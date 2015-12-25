""" calling R code from python
too lazy to write the same code in python again ;)
"""

#! /usr/bin/env python

import readline
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage
from rpy2.robjects.packages import importr

from rpy2.robjects import pandas2ri
import  rpy2.robjects.numpy2ri 

pandas2ri.activate()

import pandas as pd


string = """
library(e1071)
# library(dplyr)


# Finds controls matching the cases as good as possible
#AS
match_controls = function(df, variable_of_interest, x1, x2, match_type){
  if (variable_of_interest == "AS"){
    d = subset(df, control ==1 | (ADOS_GOTHAM_SEVERITY > x1 & ADOS_GOTHAM_SEVERITY <= x2))
    
    if (match_type == 1){
      m = matchControls(control ~ age, data=d, contlabel = 1)
      }
    else if (match_type == 2){
      m = matchControls(control ~ age + VIQ, data=d, contlabel = 1)
    }
    else if (match_type == 3){
      m = matchControls(control ~ age + VIQ + PIQ + FIQ, data=d, contlabel = 1)
    }
    else{
      m = NA
    }
  }
    
  else if (variable_of_interest == "VIQ"){
    d = subset(df, VIQ > x1 & VIQ <= x2)
    
    if (sum(d$control == 1) > sum(d$control == 0)){ # nTDC>nASD
      contlabel = 1}
      else{
        contlabel = 0
      }
    
    if (match_type == 1){
      m = matchControls(control ~ age, data=d, contlabel = contlabel)
    }
    else if (match_type == 2){
      m = matchControls(control ~ age + PIQ, data=d, contlabel = contlabel)
    }
    else if (match_type == 3){
      m = matchControls(control ~ age + PIQ + FIQ, data=d, contlabel = contlabel)
    }
    else{
      m = NA
    }
  }
  else{
    m = NA
  }

    return(m)
}


"""

powerpack = SignatureTranslatedAnonymousPackage(string, "powerpack")


df = pd.read_csv('../data/FSS.csv')

a = powerpack.match_controls(df, 'AS', 0 , 3, 1)
print(a)

