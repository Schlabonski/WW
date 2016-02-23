#include <omp.h>
#include <boost/program_options.hpp> 
#include <iostream>
#include <fstream>
#include <iterator>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <vector>

/*
This tool is used to fit a simple rate equation on the NEQ simulation of the villin.
To compile the source code:
g++ rate_est_H2O_fab.cpp -o ../bin/rate_est_H2O_fab -lboost_program_options -lm -fopenmp -std=c++11
*/

/*TODO:
 * - replace the occurences of residue 15 with heater residue 19
 * - rewrite function for the calculation of rates so that polar rates are calculated as well
 * - replace individual fitting of rates by fitting with only two diffusion constants for polar and 
 *   backbone contacts
 * - which input is no longer needed? which input has to be added?
 * - currently commented/read until L166
*/

namespace po = boost::program_options;
using namespace std; 
//tr: model trjectory
//k: rate matrix
//rt: refernce trajectory
void trj_calc(vector<vector<float>> &tr, vector<vector<float>> &k, vector<vector<float>> &rt, int &nrr, int steps, int startt){
  float sum;
  int i,j,n;
  for(i=1;i<steps;i++){
    tr[i][0] = tr[0][0]+i*0.002;
    #pragma omp parallel for private(sum,j) shared(tr,k,i)
      for(j=0;j<nrr;j++){
        sum=tr[i-1][j+1];
        #pragma omp parallel for shared(j,tr,k,i) private(n) reduction(+:sum)
          for(n=0;n<nrr;n++){
            if(j!=n){
              sum+=-(k[j][n])*tr[i-1][j+1]+(k[n][j])*tr[i-1][n+1];
            }
          }
        sum-=(k[nrr][nrr])*tr[i-1][j+1]-(k[nrr+1][nrr+1])*(403.29+211.95-rt[startt+i][nrr+1]); //solvent energy, and solvent energy at initial fitting time
        tr[i][j+1]=sum;
      }
  }
}

void var_calc(vector<vector<float>> &tr, vector<vector<float>> &rt, int startt, int steps, int &nrr, float &varres){
  float var=0;
  int i,n,count=0;
  #pragma omp parallel for private(n,i) shared(tr,rt,startt,steps,nrr) reduction(+:var,count)
  for(i=0;i<steps;i++){
    for(n=1;n<nrr;n++){
      //splitted heater is not taken into account in variance calculations
      if(n!=16 && n!=17){
        var+=(tr[i][n]-rt[startt+i][n])*(tr[i][n]-rt[startt+i][n])/rt[startt+i][n]/rt[startt+i][n];
        count++;
      }
    }
  }
  varres=var/count;

}


int main(int ac, char* av[])
{
  try{
    int xdrOK;
    int i=0, j=0, n=0;
    int stepnr;
    float temp,t=50.;
    float split=0.652455;
    vector< vector < float > > reft;
    string fi,fn,fr,fo,df,rf,to,xf;
    ostream* fp = &cout;
    ofstream rout,tout;
    fstream fnf,doff,orf,inf,ratf, dxf;
    po::options_description desc("Allowed options");
    desc.add_options()
      ("help", "produce help message")
      ("input-file,f", po::value<string>(&fi), "Kinetic energy input file")
      ("rate-file,k", po::value<string>(&fr), "File for writing the trans rates")
      ("trj-file,t", po::value<string>(&to), "File for writing the energy trajectory")
      ("index-file,n", po::value<string>(&fn), "File containing the residue pairs")
      ("dof-file,d", po::value<string>(&df), "File containing the DOF")
      ("rati-file,r", po::value<string>(&rf), "File containing the first rate estimates")
      ("dx-file,x", po::value<string>(&xf), "File containing the average distance");

    po::variables_map vm;
    po::store(po::parse_command_line(ac, av, desc), vm);
    po::notify(vm);    

    if(vm.count("help")) {
      cout << desc << "\n";
      return 1;
    }
      
    if(vm.count("rate-file")) {
      cout << "Will use  " << fr << " for writing.\n";
      rout.open(fr.c_str());
      fp =&rout;
    }
    
    if(vm.count("trj-file")){
      tout.open(to.c_str());
    }
    else{
      cout << "Dodn't know where to write the trajectory\n";
      exit(EXIT_FAILURE);
    }

    if(vm.count("input-file")){
      fnf.open(fi.c_str());
    }
    else{
      cout << "Without reference: nothing to be done\n";
      exit(EXIT_FAILURE);
    }

    if(vm.count("index-file")){
      inf.open(fn.c_str());
    }
    else{
      cout << "You did not specify any atoms.\n";
      exit(EXIT_FAILURE); 
    }

    // fill up a vector with the indices of the residues
    vector <vector< int >> ind;

    while(1){
      inf >> i >> j;
      if(inf.eof()) break;
      vector <int> newInd (2);
      newInd[0]=i;
      newInd[1]=j; 
      ind.push_back(newInd);
    }
    
    int nrcontsI = ind.size(); //number of contacts
    cout << nrcontsI << endl;

    if(vm.count("dof-file")){
      doff.open(df.c_str());
    }
    else{
      cout << "No DOF specified\n";
      exit(EXIT_FAILURE);
    }

    // fill a vector with the degrees of freedom of each residue
    vector < int > dofa;

    while(true){
      doff >> i;
      if(doff.eof()) break;
      dofa.push_back(i);
    }

    int nres=dofa.size();
    int cols=nres+1; // split heater in two, so one collumn more! 
    cout << "NR res: " << nres << "\n"; 

    if(vm.count("rati-file")){
      cout << nres << "\n";
      ratf.open(rf.c_str());
    }
    else{
      cout << nres << "\n";
      cout << "Bring me the first estimates\n";
      exit(EXIT_FAILURE);
    }
    float k0,start15,start17;
    cout << "Before k0 read\n";
    vector < vector <float> > k (nres+2, vector<float>(nres+2,0));
    ratf >> temp >> split >> k0 >> start15 >> start17 >> k[nres][nres] >> k[nres+1][nres+1]; //last diviation between ref and model, k0 for the backbone, neighboring res of heater start values, solvent loss and backrate
    //reading first estimates for the contact transport
    for(n=0;n<nrcontsI;n++){
      ratf >> i >> j >> temp;
      if(ratf.eof()){
        cout << "Not enouth rates given\n";
        exit(EXIT_FAILURE);
      }
      k[i-1][j-1]=temp;
      if(i != 16 && i != 17 && j != 16 && j != 17){
        k[j-1][i-1]=k[i-1][j-1]*dofa[i-1]/dofa[j-1];
        cout << k[j-1][i-1] << endl;
      }
    }

    if(vm.count("dx-file")){
      dxf.open(xf.c_str());
    }
    else{
      cout << "Need the average distances\n";
      exit(EXIT_FAILURE);
    }
    //here we start calculating the rate matrix
    //TODO: add calculation of polar contact rates! 
    while(true){
      dxf >> i >> j >> temp;
      if(dxf.eof()) break;
        if(i<15){ 
          //two beats for the heater with index 15 and 16
          k[i-1][j-1]=k0/(temp*temp)*sqrt(dofa[j-1]/float(dofa[i-1]));
          k[j-1][i-1]=k0/(temp*temp)*sqrt(dofa[i-1]/float(dofa[j-1]));
        }
        else{
          cout << i << "  " << j << "  " << temp << endl;
          k[i][j]=k0/(temp*temp)*sqrt(dofa[j]/float(dofa[i]));
          k[j][i]=k0/(temp*temp)*sqrt(dofa[i]/float(dofa[j]));
        }
      
    }

    while(true){//reading ref. trajectory
      fnf >> temp;
      if(fnf.eof()) break;
      vector <float> newColumn (cols+1);
      newColumn[0]=temp;
      for(i=0;i<nres+1;i++){
        fnf >> temp;
        if(i==15){
          newColumn[i+1]=temp*split;
          newColumn[i+2]=temp*(1-split);
          i++;
        }
        else{
          newColumn[i+1]=temp;
        }
      }
      reft.push_back(newColumn);
    }
    
    //cout << reft[0][16] << "  " << reft[0][17] << "\n";

    int refsteps = reft.size();
    int tsteps = int((t-0.2)/(0.002))+1;//number of fitting steps 
    int tstart = int((0.2-0.1)/0.002);//start index for fitting


    vector <vector<float>> trj (tsteps, vector<float> (cols,0));

    for(i=0;i<cols;i++){
      trj[0][i]=reft[tstart][i];//start values for the model trj
    }
    
    trj_calc(trj, k, reft, nres,tsteps, tstart);
   
    float var,prev_var;
    var_calc(trj, reft, tstart, tsteps, cols, prev_var);
    cout << prev_var << "\n";
    float times[1] = {50.};//last timstep for the fit
    int nr_times = 1;//number of last times
    float dka[1] = { 0.002}; //fraction by which the parameters are shifted
    int nr_dka = 1;
    float limits[1] = {0.00000000001}; //variance one wants to reach
    int nr_limits = 1;
    float si,sw=0;
    int count,m;
    bool isnew=false;
    float prev_k;
    int q=0,o=0;

    trj[0][15]=start15; 

    // the main fit routine starts here
    for(n=0;n<nr_times;n++){
      tsteps = int((times[n]-0.2)/(0.002))+1;
      trj.resize(tsteps, vector<float> (cols,0));
      for(i=0;i<cols;i++){
        trj[0][i]=reft[tstart][i];
      }
      trj[0][15]=start15;
      trj[0][18]=start17;
      trj_calc(trj, k, reft, nres,tsteps, tstart);
      var_calc(trj, reft, tstart, tsteps, cols, prev_var);
      for(i=0;i<nr_dka;i++){
        for(j=0;j<nr_limits;j++){
          m=0;
          ++o;
          while(1){
            si=1.;
            count=0;
            while(1){
              if(m>=nrcontsI){//fit of solvent back rate
                if(m==nrcontsI+5){
                  prev_k=k[nres+1][nres+1];
                  k[nres+1][nres+1]+=si*k[nres+1][nres+1]*dka[i];
                }
                if(m==nrcontsI+4){//fit of solvent loss rate
                  prev_k=k[nres][nres];
                  cout << "Water: " << k[nres][nres] << endl;
                  k[nres][nres]+=si*k[nres][nres]*dka[i]/100.;
                }
                if(m==nrcontsI+3){//fit of res 17 start value
                  prev_k=trj[0][18];
                  trj[0][18]+=si*dka[i]*trj[0][18];
                }
                if(m==nrcontsI+2){//fit of res 15 start value
                  prev_k=trj[0][15];
                  if(prev_k+si*dka[i]*trj[0][15]>4.){  
                    trj[0][15]+=si*dka[i]*trj[0][15];
                  }
                }
                if(m==nrcontsI+1){//fit of the splitting parameter of the excitation energy
                  prev_k=split;
                  split+=si*split*dka[i];
                  if(split<=1.){
                    #pragma omp parallel for private(n) shared(reft,refsteps)
                    for(n=0;n<refsteps;n++){
                      reft[n][16]=(reft[n][16]+reft[n][17])*split;
                      reft[n][17]=(reft[n][16]+reft[n][17])*(1-split);
                    }
                    trj[0][16]=reft[tstart][16];
                    trj[0][17]=reft[tstart][17];
                  }
                  else{
                    split=prev_k;
                  }
                }
              }
              else{//fit of the contact rates
                  prev_k=k[ind[m][0]-1][ind[m][1]-1];
                  k[ind[m][0]-1][ind[m][1]-1]+=si*k[ind[m][0]-1][ind[m][1]-1]*dka[i];
                  //cout << m << "  " << prev_k << " " << k[ind[m][0]-1][ind[m][1]-1] << endl;
                  if(k[ind[m][0]-1][ind[m][1]-1]>0.01 && abs(ind[m][0]-ind[m][1])!=1){
                    k[ind[m][0]-1][ind[m][1]-1]=prev_k;
                    break;
                  }
                  if(k[ind[m][0]-1][ind[m][1]-1]>0.2 && abs(ind[m][0]-ind[m][1])==1){
                    k[ind[m][0]-1][ind[m][1]-1]=prev_k;
                    break;
                  }
                  if(ind[m][0]!=16 && ind[m][0]!=17 && ind[m][1]!=16 && ind[m][1]!=17){
                    k[ind[m][1]-1][ind[m][0]-1]=k[ind[m][0]-1][ind[m][1]-1]*dofa[ind[m][0]-1]/dofa[ind[m][1]-1];
                    //cout << k[ind[m][1]-1][ind[m][0]-1]  <<"  shift\n";
                  }
              }
              trj_calc(trj, k, reft, nres,tsteps,tstart);
              var_calc(trj, reft, tstart, tsteps, cols, var);
              //resetting the rates if the fit got worse
                if(prev_var<var){
                  if(m>=nrcontsI){
                    if(m==nrcontsI+5){
                      k[nres+1][nres+1]=prev_k;
                    }
                    if(m==nrcontsI+4){
                      k[nres][nres]=prev_k;
                    }
                    if(m==nrcontsI+3){
                      trj[0][18]=prev_k;
                    }
                    if(m==nrcontsI+2){
                      trj[0][15]=prev_k;
                    }


                    if(m==nrcontsI+1){
                      split-=si*split*dka[i];
                      #pragma omp parallel for private(n) shared(reft,refsteps)
                      for(n=0;n<refsteps;n++){
                        reft[n][16]=(reft[n][16]+reft[n][17])*split;
                        reft[n][17]=(reft[n][16]+reft[n][17])*(1-split);
                      }
                      trj[0][16]=reft[tstart][16];
                      trj[0][17]=reft[tstart][17];
                    }
                  }
                  else{
                      k[ind[m][0]-1][ind[m][1]-1]=prev_k;
                      if(ind[m][0]!=16 && ind[m][0]!=17 && ind[m][1]!=16 && ind[m][1]!=17){
                        k[ind[m][1]-1][ind[m][0]-1]=prev_k*dofa[ind[m][0]-1]/dofa[ind[m][1]-1];
                      }
                  }
                  if(count==0){
                    si=-1.;
                  }
                  else{
                    if(count>1){
                      sw++;
                    }
                    break;
                  }
                }
                else{
                  sw++;
                  if((prev_var-var)<prev_var*limits[j] || count>10){
                    prev_var=var;
                    break;
                  }
                  prev_var=var;
                }
                count++;
            }
            m++;
            if(m==nrcontsI+7){
              if(sw>0){
                m=0;
                ++o;
                sw=0;
              }
            }
            if(m>=nrcontsI+7 || prev_var<0.0001 || o>2){
              break;
            }
            cout << m << " " << o << "  " << times[n] << " " << dka[i] <<" "<< k[14][13] <<" "<< prev_var << " " << split << " " << trj[0][15] << " " << k0 << "\n";
          }
          m=0;
          sw=0;
          if(prev_var<0.01){
            break;
          }
        }
        if(prev_var<0.01){
          break;
        }
      }
      if(prev_var<0.01){
        break; 
      }
    }

    //cout << prev_var << "\n"; 
    *fp << var << "  " << split << " " << k0 << " " << trj[0][15] << " " << trj[0][18] << " " << k[nres][nres]  << "  " << k[nres+1][nres+1] << " " << "\n";
    //cout << nrcontsI << "\n";
    for(i=0;i<nrcontsI;i++){
      *fp << ind[i][0] <<"  "<< ind[i][1] <<"  "<< k[ind[i][0]-1][ind[i][1]-1] << "\n";
    }

    //*fp.close();

    trj_calc(trj, k, reft, nres,tsteps,tstart);
    
    for(i=0;i<tsteps;i++){
      tout << trj[i][0];
      for(j=1;j<cols;j++){
        tout <<"  " << trj[i][j];
      }
      tout << "\n";  
    }

    tout.close();
    
  }
  catch(exception& e) {
    cerr << "error: " << e.what() << "\n";
    return 1;
  }
    catch(...) {
    cerr << "Exception of unknown type!\n";
  }
  return 0;
}
