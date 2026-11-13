"""Fit the first new-data RCS fold early to validate the complete RCS path."""
from run import fit_fold

if __name__=='__main__':
    print(fit_fold(1,1),flush=True)
