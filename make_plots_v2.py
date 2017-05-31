from ROOT import TFile, TCanvas, TGraph, TMultiGraph
from ROOT import gStyle, gPad, gDirectory
from ROOT import TH1F, TLegend, TPad, TColor
from ROOT import kBlue, kRed, kOrange, kYellow, kGreen, kMagenta,\
                 kCyan, kAzure, kGray, kBlack, kViolet
from os import listdir
from os.path import join
import ROOT
import collections

import numpy as np
import matplotlib.pyplot as plt

# Work from home?
WFH = False

# Get all ntuples to be plotted
if WFH: path = "/media/sdb1/HEP_DATA/IsoVars/"
else: path = "/lhcb5/users/gerstel/analysis/"
root_file = "DVNtuple.root"
# all decay modes
#modes = listdir(path)
# modes in good order
#modes = ['Bs_mutau,pipipinu', 'Bd_mutau,pipipinu', 'Bd_D-3pi,Kmunu', 'Bd_D-TauNu,munu,3pinu', 'Bd_D-munu,3pipi0',
#        'Bd_D-TauNu,3pipi0,munu', 'Bd_D-3pi,munu', 'Bd_Dst-3pipi0,munu', 'Bd_Dst-TauNu,3pipi0,munu',
 #       'Bd_Dst-3pi,munu', 'Bd_Dst-munu,3pipi0', 'Bu_DststTauNu,munu,3pinu', 'Bu_Dststmunu,3pipi0',
  #      'Bu_DststTauNu,3pipi0,munu', 'Bd_Dststmunu,3pipi0', 'Bs_Dsmunu,TauNu', 'Bs_Ds3pi,munu']

# Group modes to categories; may be convenient, even if flat list of all modes
# will be used.
modes = [('Dstst', ['Bd_Dststmunu,3pipi0', 'Bu_Dststmunu,3pipi0', 'Bu_DststTauNu,3pipi0,munu', 'Bu_DststTauNu,munu,3pinu']),
         ('other', ['Bd_Dst-3pi,munu',
                       'Bd_D-3pi,munu', 'Bd_D-3pi,Kmunu', 'Bd_Dst-3pipi0,munu', 'Bs_Ds3pi,munu', 'Bd_Dst-munu,3pipi0', 'Bd_D-munu,3pipi0',
                       'Bd_D-TauNu,3pipi0,munu', 'Bd_Dst-TauNu,3pipi0,munu', 'Bd_D-TauNu,munu,3pinu', 'Bs_Dsmunu,TauNu'])]
modes = collections.OrderedDict(modes)

# Get a flat list of all modes
flatten = lambda l: [item for sublist in l for item in sublist]
mode_list = [m for m in flatten(modes.values())]

# Add new signal modes from Julien:
mode_list += ["%s_mutau,pipipinu_tauola_babar_%s"%(part, vintage) for part in ['Bd', 'Bs'] for vintage in ['2011', '2012', '2011-12']]
print "MODES: ", mode_list


# Get all ROOT file; each mode corresponds to directory with ROOT file
fnames = [join(path, direc, root_file) for direc in mode_list]

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

palette = [1,2,3,4,5,6,7,8,12,27,20,38,9]
#palette = [kBlack, kRed, kBlue, kOrange, kViolet]

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
# important_old_leaves = ['B_SmallestDeltaChi2MassOneTrack', 'B_CDFIso_NEW',
#                         #'B_0_50_cc_mult', # not found
#                         #'B_0_50_nc_sPT' , # not found
#                        'Mu_isolation_Giampi_nopi', 'Mu_BDTiso3',
#                        #'Mu_BDTiso1_1') [where BDTiso1 = BDTiso1_1 + BDTiso1_2*100 + BDTiso1_3 *10000],
#                         'Tau2Pi_SmallestDeltaChi2MassTwoTracks', 'Tau2Pi_iso2',
#                         'Tau2Pi_BDTiso3']
#                        #'Tau2Pi_BDTiso1_1',
# # part = "SumPis"
# # '%s_BDTiso3') [is Tau2Pi_Pi1_BDTiso3 + Tau2Pi_Pi2_BDTiso3 + Tau2Pi_Pi3_BD'%s_BDTiso1_1')

###
important_old_leaves = ['B_SmallestDeltaChi2OneTrack'
                        ,'B_CDFIso_NEW'
                        ,'Mu_isolation_Giampi_nopi'
                        ,'Mu_BDTiso3'
                        ,'Tau2Pi_SmallestDeltaChi2OneTrack'
                        ,'Tau2Pi_BDTiso3']

# Ranges of hists
leaves_ranges = {'B_SmallestDeltaChi2OneTrack'      : (None, 3000), # there are events in range 3000-8000
                 'B_CDFIso_NEW'                     : (None, None),
                 'Mu_isolation_Giampi_nopi'         : (None, None),
                 'Mu_BDTiso3'                       : (-1.0, 0.01),
                 'Tau2Pi_SmallestDeltaChi2OneTrack' : (None, 1500),
                 'Tau2Pi_BDTiso3'                   : (-1.5, 0.01)}

# roughly best cppm's variables
#nominal = ['Tau2Pi_BDTiso3']

print "\n\nIsolation leaves: ", iso_leaves

my_leaves = [l for l in iso_leaves if "Dawid" in l]
not_my_leaves = [l for l in iso_leaves if not "Dawid" in l]

# Group my_leaves according to input var set
CONFIG = [str(i) for i in [1,3]]
# leaves grouped according to input variable configuration
same_var_l = [[l for l in my_leaves if "var"+config in l] for config in CONFIG ]
print "** SAME VAR LEAVES: ", same_var_l
# min_{all-tracks}(BDT) for each input var set
min_bdtg_per_evt =["min(min(" + ",".join(svl[:2]) + "),min(" + \
    ",".join(svl[2:4]) + "))" for svl in same_var_l]

for el in min_bdtg_per_evt:
    leaves_ranges[el] = (None, None)

# "New" variables so that we have all Joan's ones (as in his NOTE)
sumPi_BDTiso3 = '+'.join(["Tau2Pi_Pi%d_BDTiso3"%(id) for id in [1,2,3]])
sumPi_BDTiso1_1 = '+'.join(["Tau2Pi_Pi%d_BDTiso1-int(Tau2Pi_Pi%d_BDTiso1/100)*100"%(id,id) for id in [1,2,3]])
decoded = ["%s_BDTiso1-int(%s_BDTiso1/100)*100"%(part,part) for part in ["Tau2Pi", "Mu"]]
new = [sumPi_BDTiso3, sumPi_BDTiso1_1] + decoded

for el in new:
    leaves_ranges[el] = (None, None)

print "*** RANGES OF LEAVES: ", leaves_ranges

# Aliases to leaves, so that legend display is shorter
leaves_alias = {sumPi_BDTiso3 : 'sumPi_BDTiso3',
                sumPi_BDTiso1_1 : 'sumPi_BDTiso1_1',
                decoded[0] : 'Tau2Pi_BDTiso1_1',
                decoded[1] : 'Mu_BDTiso1_1',
                min_bdtg_per_evt[0] : 'BDT_Dawid_1',
                min_bdtg_per_evt[1] : 'BDT_Dawid_3'}

def getLeafName(i):
  """ Get alias if exists, otherwise full name """
  if leaves_subset[i] in leaves_alias.keys():
    return leaves_alias[leaves_subset[i]]
  return leaves_subset[i]

######################################################################
# CONTROLS
######################################################################
leaves_subset = min_bdtg_per_evt + important_old_leaves + new #not_my_leaves[:2] #min_bdtg_per_evt
#leaves_subset = [leaves_subset[i] for i in [0,1,2,8]]

if WFH: OUTDIR = "./plots/"
else: OUTDIR = "/lhcb/users/gerstel/LHCb_Analysis/DaVinciDev_v37r2p4/plots/"
OUTFILE = OUTDIR + "old_iso_vars.pdf"

# Apply preselection cuts (Joan's NOTE)
#CUTS = ""
CUTS = "(B_BDFplus_init_st==0)&&(B_BDFplus_status==0)&&(B_BDFplus_M>3500.)&&(B_BDFplus_M<7000.)"
cut = "(Tau2Pi_SmallestDeltaChi2MassOneTrack-Tau2Pi_M<1000)"
CUTS = '&&'.join([CUTS, cut])

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
  """ Take sig and bkg TH1F histograms and compute their ROC curve
      Assume sig,bkg are normalised!

      TODO: flexible 'reversed' way (if sig has low bdts, e.g.)
   """

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
    return 1 - self.bkg.Integral(i, self.Nbins, "width")

  def roc(self):
    """ Compute ROC curve, retrieve it inside class """
    # Include under/over -flow bins
    xx = np.array([self.s_eff(i) for i in range(0, self.Nbins+2)])
    yy = np.array([self.b_rej(i) for i in range(0, self.Nbins+2)])
    #print "xx: ", xx, "\n\nyy: ", yy
    self.roc_curve = TGraph(len(xx), xx, yy)
    self.roc_curve.Draw("AC*")
#    self.roc_curve = (xx, yy)
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
    self.canvases = []
    self.merged_canvases = [TCanvas() for l in leaves_subset]

  def __call__(self):
    self.make_raw_hists()
    self.select_sig()
    self.select_bkg()
    self.merge_signals()
    self.merge_backgrounds()
    self.plot_sig_and_bkg()
    self.make_ROCs()
    self.draw_corr_B_mass()

  def make_raw_hists(self):
    # Open canvas and output file
    C = TCanvas()
    C.Print("plots/output.pdf[")
    C.Print("plots/output.root[")
    self.canvases = [TCanvas() for l in leaves_subset]

    for i,l in enumerate(leaves_subset):
      self.plot_my_leave(leaf_id=i, ymax_all=None, xmax_all=leaves_ranges[l][1])

    # Close the output file
    C.Print("plots/output.root]")
    C.Print("plots/output.pdf]")

  def select_sig(self):
    self.sig_h = [[self.hists[i][j] for i in range(len(mode_list)) \
                   if "Bs_mutau,pipipinu" in self.hists[i][j].GetName() or \
                   "Bd_mutau,pipipinu" in self.hists[i][j].GetName()] for j in \
                   range(len(leaves_subset))]
    print "sig hists: ", self.sig_h


  def select_bkg(self):
    self.bkg_h = [[self.hists[i][j] for i in range(len(mode_list)) \
                     if "Dstst" in self.hists[i][j].GetName()] for j in \
                     range(len(leaves_subset))]
    print "\nbkg hists: ", self.bkg_h


  def add_hists(self, h_list, label, title):
    his0 = h_list[0]
    nbins = his0.GetNbinsX()
    xmax = his0.GetXaxis().GetXmax()
    xmin = his0.GetXaxis().GetXmin()
    res = TH1F(label, title, nbins, xmin, xmax)
    for i in range(1, h_list[0].GetNbinsX()+1):
      bin_centre = h_list[0].GetBinCenter(i)
      res.Fill(bin_centre, sum(h.GetBinContent(i) for h in h_list))
    return res

  def normalise(self, h, verbose=False):
    """ Including under/over flow """
    if verbose: print "*** Before normalise, under/over -flow: ", h.GetBinContent(0), \
                h.GetBinContent(h.GetNbinsX()+1)
    norm = h.Integral(0, h.GetNbinsX()+1, "width")
    if verbose: print "Before normalising hist: ", h, " its norm: ", norm
    h.Scale(1. / norm)
    if verbose:
      print "After... ", h.Integral("width")
      print "*** After normalise, under/over -flow: ", h.GetBinContent(0), \
          h.GetBinContent(h.GetNbinsX()+1)

  def merge_signals(self):
    self.sig_merged = [self.add_hists(self.sig_h[:][i], label="sig_"+l, \
                       title="Merged "+getLeafName(i)) for i,l in enumerate(leaves_subset)]
    print "MERGED SIG: ", self.sig_merged
    for s in self.sig_merged: self.normalise(s)

  def merge_backgrounds(self):
    self.bkg_merged = [self.add_hists(self.bkg_h[:][i], label="bkg_"+l, \
                       title="Merged "+getLeafName(i)) for i,l in enumerate(leaves_subset)]
    print "MERGED BKG : ", self.bkg_merged
    for s in self.bkg_merged: self.normalise(s)

  def plot_sig_and_bkg(self):
    C = TCanvas()
    C.Print("plots/sig_and_bkg.pdf[")
    for i in range(len(leaves_subset)):
      self.merged_canvases[i].Divide(1,1)
      self.merged_canvases[i].cd(1)
      #self.merged_canvases.append(TCanvas("c", "Sig,Bkg for %s"%(i)))
      self.sig_merged[i].SetLineColor(kBlue)
      self.sig_merged[i].Draw("HIST")
      self.bkg_merged[i].SetLineColor(kRed)
      self.bkg_merged[i].Draw("HIST same")
      self.merged_canvases[i].Update()
      self.merged_canvases[i].Print("plots/sig_and_bkg.pdf")
    C.Print("plots/sig_and_bkg.pdf]")


  def make_ROCs(self):
    self.roc_curves = []
    for sig,bkg in zip(self.sig_merged, self.bkg_merged):
      roc = ROC(sig, bkg)
      self.roc_curves.append(roc())

    # Draw them
    self.cnv = TCanvas()
    self.mg = TMultiGraph("mg", \
              ";Signal efficiency;Background rejection")
    for i,rc in enumerate(self.roc_curves):
      rc.SetLineColor(palette[i])
      if i == 9:
        rc.SetMarkerColor(i+5)
      else:
        rc.SetMarkerColor(palette[i])
      rc.SetMarkerStyle(20)
      rc.SetName(leaves_subset[i])
      # if i == 0 or i == 1:
      #   rc.SetTitle("myBDT%d"%(i+1))
      # elif i == 2:
      #   rc.SetTitle("nominal")
      # else:
      #   rc.SetTitle(leaves_subset[i])
      rc.SetTitle(getLeafName(i))
      self.mg.Add(rc)

    self.mg.GetXaxis()#.SetTitleSize(10)
    self.mg.GetYaxis()#.SetTitleSize(10)
    gStyle.SetLegendBorderSize(0)
    gStyle.SetLegendTextSize(0.04)
    gStyle.SetLabelSize(.05, "XY")
    gStyle.SetTitleFontSize(.08)
    gStyle.SetTitleFont(72, "XY")
    gStyle.SetTitleXSize(0.07)
    gStyle.SetTitleYSize(0.07)
    gStyle.SetTitleXOffset(0.75)
    gStyle.SetTitleYOffset(0.7)
    gStyle.SetFrameBorderSize(5)

    gPad.SetLeftMargin(0.11)
    #gPad.SetRightMargin(0.1)
    gPad.SetBottomMargin(.119)
    #gPad.SetTopMargin(0.1)

    self.mg.Draw("APL")
    self.cnv.BuildLegend(0.03, 0.03, 0.4, 0.4, "", "LP")
    self.cnv.Update()
    self.cnv.Print("plots/roc.pdf")


  def draw_corr_B_mass(self):
    self.corr_B_mass_cnv = TCanvas()
    self.corr_B_mass_cnv.Divide(6,4)

    self.hist_B_mass_corr = [TH1F(m, m, 100, 2000, 20000) for m in mode_list]
    gStyle.SetOptStat(111111)

    for i,mode in enumerate(mode_list):
      self.corr_B_mass_cnv.cd(i+1)
      trees[i].Draw("B_BDFplus_M >> " + mode, CUTS)


  def plot_my_leave(self, leaf_id, ymax_all=None, xmax_all=None):
    self.canvases[leaf_id].Divide(6, 3)
    bdtg_label = []
    legs = [TLegend(0.2,0.5,0.6,0.9) for m in mode_list]

    # Global (full canvas) xmin and xmax for alignment
    xmin_g, xmax_g = (0, 0)
    mode_id = -1
    for mode in mode_list:
      mode_id += 1
      self.canvases[leaf_id].cd(mode_id + 1) #group_id+1)
      bdtg_label.append(get_hist_label(leaf_id, mode_id))
      draw_cmd = leaves_subset[leaf_id]+">>"+bdtg_label[-1]
      trees[mode_id].Draw(draw_cmd, CUTS, "goff")
      htmp = gDirectory.Get(bdtg_label[-1])

      # Update minima/maxima
      xmin, xmax = (htmp.GetXaxis().GetXmin(), htmp.GetXaxis().GetXmax())
      if xmin < xmin_g: xmin_g = xmin
      if xmax > xmax_g: xmax_g = xmax

      print "xmax_all = ", xmax_all

    print "Leaf%d: xmin_g, xmax_g = %f, %f"%(leaf_id, xmin_g, xmax_g)
    # Draw the leaves into the histograms
    # Their same binning and ranges help in making the ROC curve
    for mode_id, mode in enumerate(mode_list):

      # Define histogram
      label = mode+str(leaf_id)#leaves_subset[leaf_id]
      if xmax_all is not None:
        self.hists[mode_id].append(TH1F(label, getLeafName(leaf_id), 1000, -1.0, xmax_all))
      else:
        self.hists[mode_id].append(TH1F(label, getLeafName(leaf_id), 1000, xmin_g, xmax_g))
      if ymax_all is not None:
        self.hists[mode_id].SetMaximum(ymax_all)

      # Draw histogram with legend
      self.canvases[leaf_id].cd(mode_id + 1)
      draw_cmd = leaves_subset[leaf_id]+">>"+label
      trees[mode_id].Draw(draw_cmd, CUTS, "HIST")
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
    self.canvases[leaf_id].Update()  # Important also for axes alignment
    self.canvases[leaf_id].Print("plots/output.root")
    self.canvases[leaf_id].Print("plots/output.pdf")

######################################################################
# MAIN
######################################################################
isoAn = IsolationAnalyser()
isoAn()
