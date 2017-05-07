from ROOT import TFile, TCanvas
from ROOT import gStyle, gDirectory
from ROOT import TH1F, TLegend, TPad, TColor
from ROOT import kBlue, kRed, kOrange, kYellow, kGreen, kMagenta,\
                 kCyan, kAzure, kGray, kBlack, kViolet
from os import listdir
from os.path import join
import ROOT
import collections

import numpy as np

# Get all ntuples to be plotted
#path = "/media/sdb1/HEP_DATA/IsoVars/"
path = "/lhcb5/users/gerstel/analysis/"
root_file = "DVNtuple.root"
# all decay modes
#modes = listdir(path)
# modes in good order
#modes = ['Bs_mutau,pipipinu', 'Bd_mutau,pipipinu', 'Bd_D-3pi,Kmunu', 'Bd_D-TauNu,munu,3pinu', 'Bd_D-munu,3pipi0',
#        'Bd_D-TauNu,3pipi0,munu', 'Bd_D-3pi,munu', 'Bd_Dst-3pipi0,munu', 'Bd_Dst-TauNu,3pipi0,munu',
 #       'Bd_Dst-3pi,munu', 'Bd_Dst-munu,3pipi0', 'Bu_DststTauNu,munu,3pinu', 'Bu_Dststmunu,3pipi0',
  #      'Bu_DststTauNu,3pipi0,munu', 'Bd_Dststmunu,3pipi0', 'Bs_Dsmunu,TauNu', 'Bs_Ds3pi,munu']
modes = [('sig', ['Bs_mutau,pipipinu' , 'Bd_mutau,pipipinu']),
         ('D(mu)3pi', ['Bd_Dst-3pi,munu', 'Bd_D-3pi,munu', 'Bd_D-3pi,Kmunu',
                       'Bd_Dst-3pipi0,munu', 'Bs_Ds3pi,munu']),
         ('D(3pi)mu', ['Bd_Dst-munu,3pipi0', 'Bd_D-munu,3pipi0',
                       'Bd_Dststmunu,3pipi0', 'Bu_Dststmunu,3pipi0']),
         ('D(3pi)tau(mu)', ['Bd_D-TauNu,3pipi0,munu',
                            'Bd_Dst-TauNu,3pipi0,munu',
                            'Bu_DststTauNu,3pipi0,munu']),
         ('D(mu)tau(3pi)', ['Bd_D-TauNu,munu,3pinu',
                            'Bu_DststTauNu,munu,3pinu']),
         ('D(tau(3pi))mu', ['Bs_Dsmunu,TauNu'])
        ]
modes = collections.OrderedDict(modes)

flatten = lambda l: [item for sublist in l for item in sublist]
mode_list = [m for m in flatten(modes.values())]
#mode_list = [mode_list[3], mode_list[15]]

fnames = [join(path, direc, root_file) for direc in mode_list]
print "MODES: ", mode_list, "\n\n"

# Colour-code each mode #
colours = {}
#kBlack, kBlue, TColor::GetColroDark(kGreen), kRed, kViolet, kOrange, kMagenta
# 2 signals
colours['Bs_mutau,pipipinu']         = kBlue
colours['Bd_mutau,pipipinu']         = kBlack
# Bd_D
colours['Bd_D-3pi,Kmunu']            = kRed
colours['Bd_D-TauNu,munu,3pinu']     = kViolet
colours['Bd_D-munu,3pipi0']          = kBlack
colours['Bd_D-TauNu,3pipi0,munu']    = kOrange
colours['Bd_D-3pi,munu']             = kBlue
# Bd_Dst
colours['Bd_Dst-3pipi0,munu']        = kRed
colours['Bd_Dst-TauNu,3pipi0,munu']  = kViolet
colours['Bd_Dst-3pi,munu']           = kBlack
colours['Bd_Dst-munu,3pipi0']        = kOrange
# Bd_Dstst
colours['Bu_DststTauNu,munu,3pinu']  = kRed
colours['Bu_Dststmunu,3pipi0']       = kViolet
colours['Bu_DststTauNu,3pipi0,munu'] = kBlack
colours['Bd_Dststmunu,3pipi0']       = kOrange
# Bs_Ds
colours['Bs_Dsmunu,TauNu']           = kBlue
colours['Bs_Ds3pi,munu']             = kBlack


# Allocate TFiles and TTrees
files = {}
trees = {}

# Get all TFiles and TTrees
for i,f in enumerate(fnames):
  files[i] = TFile(f)
  trees[i] = files[i].Get("DecayTreeTuple/DecayTree")

print "FILES: ", files
print "\n\nTREES: ", trees


# Get list of all leaves (pertaining to isolation variables)
leaves = trees[0].GetListOfLeaves()
leaves = [l.GetName() for l in leaves]
iso_leaves = [name for name in leaves if "so" in name or "Dawid" in name]
important_old_leaves = ['B_SmallestDeltaChi2MassOneTrack', 'B_CDFIso_NEW',
                        #'B_0_50_cc_mult', # not found
                        #'B_0_50_nc_sPT' , # not found
                       'Mu_isolation_Giampi_nopi', 'Mu_BDTiso3',
                       #'Mu_BDTiso1_1') [where BDTiso1 = BDTiso1_1 + BDTiso1_2*100 + BDTiso1_3 *10000],
                        'Tau2Pi_SmallestDeltaChi2MassTwoTracks', 'Tau2Pi_iso2',
                        'Tau2Pi_BDTiso3']
                       #'Tau2Pi_BDTiso1_1',
# part = "SumPis"
# '%s_BDTiso3') [is Tau2Pi_Pi1_BDTiso3 + Tau2Pi_Pi2_BDTiso3 + Tau2Pi_Pi3_BD'%s_BDTiso1_1')

###

print "\n\nIsolation leaves: ", iso_leaves

my_leaves = [l for l in iso_leaves if "Dawid" in l]
not_my_leaves = [l for l in iso_leaves if not "Dawid" in l]

# Group my_leaves according to input var set
CONFIG = [str(i) for i in [1,3]]
same_var_l = [[l for l in my_leaves if "var"+config in l] for config in CONFIG ]
print "** SAME VAR LEAVES: ", same_var_l
# min_{all-tracks}(BDT) for each input var set

min_bdtg_per_evt =["min(min(" + ",".join(svl[:2]) + "),min(" + \
    ",".join(svl[2:4]) + "))" for svl in same_var_l]

######################################################################
# CONTROLS
######################################################################
leaves_subset = important_old_leaves[:2] #not_my_leaves[:2] #min_bdtg_per_evt

#OUTDIR = "./plots/"
OUTDIR = "/lhcb/users/gerstel/LHCb_Analysis/DaVinciDev_v37r2p4/plots/"
OUTFILE = OUTDIR + "old_iso_vars.pdf"

######################################################################
# Allocate histograms
######################################################################

#h_ bdtg = {}
# for j,l in enumerate(leaves_subset):
#   h_bdtg[j] = {}

# counter = 0
# for j in range(len(leaves_subset)):
#   for i in range(len(mode_list)):
#     counter += 1
#     print "Allocate mode: ", mode_list[i]
#     bdtg_label = "bdtg" + "j" + str(j) + "i" + str(i)
#     print "*** HIST LABEL *** : ", bdtg_label
#     title = leaves_subset[j]+';;'#+leaves_subset[j]+';'
#     h_bdtg[j][i] = TH1F(bdtg_label, title, 100, -1.0, 1.0)

######################################################################
# Superimpose hist #
######################################################################

def get_hist_label(j, i):
  """ Helper to uniquely name histograms """
  return "bdtg" + "j" + str(j) + "i" + str(i)


class ROC(object):
  def __init__(self, sig, bkg):
    self.sig = sig
    self.bkg = bkg
    assert(self.sig.GetNbinsX() == self.bkg.GetNbinsX())
    assert(self.sig.GetBinWidth(1) == self.bkg.GetBinWidth(1))
    self.Nbins = self.sig.GetNbinsX()

  def __call__(self):
    return self.roc()

  def s_eff(self, i):
    """ Signal efficiency for i-th bin """
    return self.sig.Integral(i, self.Nbins, "width")

  def b_rej(self, i):
    """ Background rejection for i-th bin """
    return self.bkg.Integral(1, i, "width")

  def roc(self):
    """ Compute ROC curve, retrieve it inside class """
    xx = np.array([self.s_eff(i) for i in range(self.Nbins)])
    yy = np.array([self.b_rej(i) for i in range(self.Nbins)])
    self.roc_curve = (xx, yy)
    return self.roc_curve


class IsolationAnalyser(object):
  """ Analyse isolation variables:
      - Assumption: Various decay-modes (seperate TFiles -> TTrees)
                    Various isolation variables (TLeaves)
      - Objective: to find best isolation variables
      - Do:
        *) Draw plain histograms
        *) Merge them to obtain one hist for signal and one for background
        *) Make ROC curves for all isolation variables and show them on
           one plot
  """
  
  def __init__(self):
    self.hists = [[] for m in mode_list]

  def __call__(self):
    self.make_raw_hists()
    self.merge_signals()
    self.merge_backgrounds()

  def make_raw_hists(self):
    xmax_l = [30000, 1.5]
    # Open canvas and output file
    C = TCanvas()
    C.Print("plots/output.pdf[")

    for i,l in enumerate(leaves_subset):
      self.plot_my_leave(leaf_id=i, ymax_all=None, xmax_all=xmax_l[i])

    # Close the output file
    C.Print("plots/output.pdf]")

  def add_hists(self, h_list, label, title):
    his0 = h_list[0]
    nbins = his0.GetNbinsX()
    xmax = his0.GetXaxis().GetXmax()
    xmin = his0.GetXaxis().GetXmin()
    res = TH1F(label, title, nbins, xmin, xmax)
    for i in range(h_list[0].GetNbinsX()+1):
      bin_centre = h_list[0].GetBinCenter(i)
      res.Fill(bin_centre, sum(h.GetBinContent(i) for h in h_list))
    return res
      

  def merge_signals(self):
    #self.sig_merged = [self.add_hists(
    pass
    #self.sig_h = [self.hists[0][i] for i,l in enumerate(leaves_subset) if l in ['Bs_mutau,pipipinu' , 'Bd_mutau,pipipinu']] 
    #self.sig_h = self.merge_hists(self.sig_h[0], self.sig_h[1], label="test", title="testnig")

  def merge_backgrounds(self):
    pass
    #self.bkg_h = [self.hists[1][i] for i,l in enumerate(leaves_subset)if "Dstst" in l] 

  def make_ROCs(self):
    for sig,bkg in zip(self.sig_h, self.bkg_h):
      roc = ROC(sig, bkg)
      xx, yy = roc()
      #plt.plot(xx, yy, label=

  def plot_my_leave(self, leaf_id, ymax_all=None, xmax_all=None):
    """
    params:
    =======
    """
    c = TCanvas()
    c.Divide(6, 3)
    bdtg_label = []
    legs = [TLegend(0.2,0.5,0.6,0.9) for group in mode_list]

    # Global (full canvas) xmin and xmax for alignment
    xmin_g, xmax_g = (0, 0)
    mode_id = -1
    for mode in mode_list:
      mode_id += 1
      print "\nSuperimpose modes: ", mode
      c.cd(mode_id + 1) #group_id+1)
      bdtg_label.append(get_hist_label(leaf_id, mode_id))
      print "*** BDTG_LABEL: ", bdtg_label[-1]
      draw_cmd = leaves_subset[leaf_id]+">>"+bdtg_label[-1]
      trees[mode_id].Draw(draw_cmd, "", "goff")
      htmp = gDirectory.Get(bdtg_label[-1])

      # Update minima/maxima
      xmin, xmax = (htmp.GetXaxis().GetXmin(), htmp.GetXaxis().GetXmax())
      if xmin < xmin_g: xmin_g = xmin
      if xmax > xmax_g: xmax_g = xmax

      print "xmax_all = ", xmax_all
      print "xmax_g = ", xmax_g

    # Draw the leaves into the histograms
    # Their same binning and ranges help in making the ROC curve
    for mode_id, mode in enumerate(mode_list):

      # Define histogram
      label = mode+leaves_subset[leaf_id]
      if xmax_all is not None:
        self.hists[mode_id].append(TH1F(label, leaves_subset[leaf_id], 100, xmin_g, xmax_all))
      else:
        self.hists[mode_id].append(TH1F(label, leaves_subset[leaf_id], 100, xmin_g, xmax_g))
      if ymax_all is not None:
        self.hists[mode_id].SetMaximum(ymax_all)

      # Draw histogram with legend
      c.cd(mode_id + 1)
      draw_cmd = leaves_subset[leaf_id]+">>"+label
      trees[mode_id].Draw(draw_cmd, "")
      # Normalise hist
      norm =    self.hists[mode_id][leaf_id].Integral("width")
      print "NORM = ", norm
      if norm:    # normalise non-empty hists
        self.hists[mode_id][leaf_id].Scale(1. / norm)
      else:
        print "Mode: ", mode_list[mode_id], " has empty hist"
        legs[mode_id].AddEntry(mode, mode_list[mode_id], 'l')
        legs[mode_id].Draw()


    # Update and save canvas to single pdf page
    c.Update()  # Important also for axes alignment
    c.Print("plots/output.pdf")

######################################################################
# MAIN
######################################################################
isoAn = IsolationAnalyser()
isoAn()




