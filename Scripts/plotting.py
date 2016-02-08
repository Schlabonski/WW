#!/usr/bin/python3


################################################################################
# Usefull functions to shorten some of my plotting routine #####################
################################################################################

import matplotlib.pyplot as plt
import seaborn as sns

def set_sns_standard(context = 'paper', font_scale=1.4, linewidth=1.5, font='serif'):
  rc_params = {'lines.linewidth':linewidth, 'text.usetex':True}
  sns.set(style='ticks', font=font, palette='Set1', context=context, font_scale=font_scale, rc=rc_params)

def remove_ticks(axe, top=True, right=True):
  if right:
    plt.tick_params(
      axis='y',          # changes apply to the x-axis
      which='both',      # both major and minor ticks are affected
      left='on',      # ticks along the bottom edge are off
      right='off',         # ticks along the top edge are off
      labelright='off') # labels along the bottom edge are off
  if top:
    plt.tick_params(
      axis='x',          # changes apply to the x-axis
      which='both',      # both major and minor ticks are affected
      bottom='on',      # ticks along the bottom edge are off
      top='off',         # ticks along the top edge are off
      labelbottom='on') # labels along the bottom edge are off
