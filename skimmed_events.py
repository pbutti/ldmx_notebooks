import uproot
import numpy as np
import awkward
import concurrent.futures
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
executor = concurrent.futures.ThreadPoolExecutor(8)
filename = '4gev_1e_tskim_v12_kaons_ldmx-det-v12_toughkskim.root'

# radius of containment for each ECal layer
radius_beam_68 = [4.73798004, 4.80501156, 4.77108164, 4.53839401, 4.73273021,
4.76662872, 5.76994967, 5.92028271, 7.28770932, 7.60723209,
9.36050277, 10.03247442, 12.14656399, 13.16076587, 15.88429816,
17.03559932, 20.32607264, 21.75096888, 24.98745754, 27.02031225,
30.78043038, 33.03033267, 37.55088662, 40.14062264, 47.95964745,
55.96441035, 66.33128366, 70.42649416, 86.68563278, 102.49022815,
119.06854141, 121.20048803, 127.5236134, 121.99024095]

# distance between each ECal layer
layer_dz = np.array([7.850, 13.300, 26.400, 33.500, 47.950, 56.550, 72.250, 81.350, 97.050, 106.150,
            121.850, 130.950, 146.650, 155.750, 171.450, 180.550, 196.250, 205.350, 221.050,
            230.150, 245.850, 254.950, 270.650, 279.750, 298.950, 311.550, 330.750, 343.350,
            362.550, 375.150, 394.350, 406.950, 426.150, 438.750])

# z-position of each ECal layer
layer_z = 240.5 + layer_dz

# list of branches we want to store from the root file
branches = []
branch_suffix = 'v3_v12'

# branchnames
ecal_branch = 'EcalRecHits_{}/EcalRecHits_{}'.format(branch_suffix, branch_suffix)
ecalSP_branch = 'EcalScoringPlaneHits_{}/EcalScoringPlaneHits_{}'.format(branch_suffix, branch_suffix)
tSP_branch = 'TargetScoringPlaneHits_{}/TargetScoringPlaneHits_{}'.format(branch_suffix, branch_suffix)
ecalVeto_branch = 'EcalVeto_{}'.format(branch_suffix)
ecalSim_branch = 'EcalSimHits_{}/EcalSimHits_{}'.format(branch_suffix, branch_suffix)

# add EcalRecHits branches
for leaf in ['xpos_', 'ypos_', 'zpos_', 'energy_']:
    branches.append('{}.{}'.format(ecal_branch, leaf))

# add EcalVeto branches
for leaf in ['showerRMS_', 'epAng_', 'passesVeto_']:
    branches.append('{}/{}'.format(ecalVeto_branch, leaf))

# add EcalSPHits and TargetSPHits branches
# Use EcalSPHits for fiducial/non-fiducial and TargetSPHits for trajectories
for leaf in ['x_', 'y_', 'z_', 'px_', 'py_', 'pz_', 'pdgID_', 'trackID_']:
    branches.append('{}.{}'.format(ecalSP_branch, leaf))
    branches.append('{}.{}'.format(tSP_branch, leaf))

# add EcalSimHits branches
for leaf in ['x_', 'y_', 'z_', 'pdgCodeContribs_', 'edepContribs_', 'incidentIDContribs_']:
    branches.append('{}.{}'.format(ecalSim_branch, leaf))

# add the eventNumber leaf
branches.append('EventHeader/eventNumber_')
#print (branches)

# use uproot and load all of the data into a dictionary: {'leaf1': [data], 'leaf2': [data], ...}
t = uproot.open(filename)['LDMX_Events']
table = t.arrays(expressions=branches, interpretation_executor=executor)

# we store all of our data into the dictionary "tree"
tree = {}
for branch in branches:
    tree[branch] = table[branch]

#print(tree.keys())

# find the number of events that pass through the vetoes
# record the event numbers of the events that pass through the vetoes
events = 0
eventNumbers = []
for event in range(len(tree['EventHeader/eventNumber_'])):
    events +=1
    eventNumbers.append(tree['EventHeader/eventNumber_'][event])

#print(events)
#print(eventNumbers)

# calculate the electron and photon trajectories

# we will store the data for the trajectory positions inside this dictionary:
trajectories = {}
b1 = []
b2 = []
b3 = []
b4 = []

# loop through all of the events
for i in range(len(tree['EventHeader/eventNumber_'])):
    tSPHits = {}
    ecalSPHits = {}
    for x in ['x_', 'y_', 'z_', 'px_', 'py_', 'pz_', 'pdgID_', 'trackID_']:
        tSPHits[x] = (table['{}.{}'.format(tSP_branch, x)])[i]
        ecalSPHits[x] = (table['{}.{}'.format(ecalSP_branch, x)])[i]
   
    # find the max pz at the target scoring plane for the recoil electron
    max_pz = 0
    r = 0
    for j in range(len(tSPHits['z_'])):
        if tSPHits['pdgID_'][j] == 11 and tSPHits['z_'][j] > 4.4 and tSPHits['z_'][j] < 4.6 and tSPHits['pz_'][j] > max_pz and tSPHits['trackID_'][j] == 1:
            max_pz = tSPHits['pz_'][j]
            r = j
    #print("Found recoil e- pz at target for event {}: {}".format(tree['EventHeader/eventNumber_'][i],max_pz))

    # find the max pz at the ecal face for the recoil electron
    max_pz_e = 0
    r_e = 0
    for j in range(len(ecalSPHits['z_'])):
        if ecalSPHits['pdgID_'][j] == 11 and ecalSPHits['z_'][j] > 239 and ecalSPHits['z_'][j] < 242 and ecalSPHits['pz_'][j] > max_pz_e and ecalSPHits['trackID_'][j] == 1:
            max_pz_e = ecalSPHits['pz_'][j]
            r_e = j
    fiducial = max_pz_e != 0
    
    if fiducial:
        E_beam = 4000
        target_dist = 241.5
        # positions and trajectory vectors of recoil e- at Ecal SP
        etraj_sp = np.array((ecalSPHits['x_'][r_e], ecalSPHits['y_'][r_e], ecalSPHits['z_'][r_e]))
        enorm_sp = np.array((ecalSPHits['px_'][r_e]/ecalSPHits['pz_'][r_e], ecalSPHits['py_'][r_e]/ecalSPHits['pz_'][r_e], 1.0))
        # positions and trajectory vectors of photon at Target SP
        pnorm_sp = np.array((-tSPHits['px_'][r]/(E_beam - tSPHits['pz_'][r]), -tSPHits['py_'][r]/(E_beam - tSPHits['pz_'][r]), 1.0))
        ptraj_sp = np.array((tSPHits['x_'][r] + target_dist*pnorm_sp[0], tSPHits['y_'][r] + target_dist*pnorm_sp[1], tSPHits['z_'][r] + target_dist))
        # store the trajectory information 
        b1.append(etraj_sp)
        b2.append(enorm_sp)
        b3.append(pnorm_sp)
        b4.append(ptraj_sp)
    else:
        etraj_sp, enorm_sp, ptraj_sp, pnorm_sp = None, None, None, None

# store the e- trajectory and its normal vector, store the photon trajectory and its normal vector
trajectories['etraj_sp'] = b1
trajectories['enorm_sp'] = b2
trajectories['pnorm_sp'] = b3
trajectories['ptraj_sp'] = b4

# get the PDG IDs of the hits
Pdg_ID = []
for event in range(len(tree['EcalRecHits_v3_v12/EcalRecHits_v3_v12.xpos_'])):
    # data from EcalSimHits
    sim_dict = {}
    for x in ['x_', 'y_', 'z_', 'pdgCodeContribs_', 'edepContribs_', 'incidentIDContribs_']:
        sim_dict[x] = table['{}.{}'.format(ecalSim_branch, x)][event]

    # data from EcalRecHits
    ecal_dict = {}
    for x in ['xpos_', 'ypos_', 'zpos_']:
        ecal_dict[x] = table['{}.{}'.format(ecal_branch, x)][event]

    rec_matched_ids = []
    rec_parent_ids = []

    for j in range(len(ecal_dict['zpos_'])):
        # For each hit:  Find a contrib w/ the same position, then match:
        simIndex = None # EcalSimHit index
        contribIndex = None # index of contribution within EcalSimHit
        for k in range(len(sim_dict['x_'])):
            if round(sim_dict['x_'][k]) == round(ecal_dict['xpos_'][j]) and \
                round(sim_dict['y_'][k]) == round(ecal_dict['ypos_'][j]) and \
                round(sim_dict['z_'][k]) == round(ecal_dict['zpos_'][j]):
                simIndex = k # we found a matching hit. 
                # now, go through the contribs and find the pdgID of the contrib w/ max edep:
                eDepMax = 0
                for l in range(len(sim_dict['edepContribs_'][k])):
                    if sim_dict['edepContribs_'][k][l] > eDepMax:
                        eDepMax = sim_dict['edepContribs_'][k][l]
                        contribIndex = l
        if not simIndex:  # If no EcalSimHit found, presumably noise; record
            rec_matched_ids.append(-9999999)
            rec_parent_ids.append(-9999999)
        else:
            rec_matched_ids.append(sim_dict['pdgCodeContribs_'][simIndex][contribIndex])
            rec_parent_ids.append(sim_dict['incidentIDContribs_'][simIndex][contribIndex])
    Pdg_ID.append(rec_matched_ids)

tree['Pdg_ID'] = Pdg_ID

# loop through each event and cut all of the hits within electron containment radii

# this is a new dictionary that will story only the hits that are outside the e- containment radii
cutTree = {}

eventsX = []
eventsY = []
eventsZ = []
pdgIDs = []

# loop through each event 
for event in range(len(tree['EcalRecHits_v3_v12/EcalRecHits_v3_v12.xpos_'])):
    ecal_front = 240.5
    etraj_front = np.array(trajectories['etraj_sp']) 
    ptraj_front = np.array(trajectories['ptraj_sp']) 

    # obtain the event's e- trajectory layer intercepts
    eLayerIntercepts = []
    intercept = (trajectories['etraj_sp'][event][0] + layer_dz*trajectories['enorm_sp'][event][0], 
                trajectories['etraj_sp'][event][1] + layer_dz*trajectories['enorm_sp'][event][1], 
                trajectories['etraj_sp'][event][2] + layer_dz*trajectories['enorm_sp'][event][2])
    eLayerIntercepts.append(intercept)
    eLayerIntercepts = np.concatenate(np.array(eLayerIntercepts))

    # obtain the event's photon trajectory layer intercepts
    pLayerIntercepts = []
    intercept = (trajectories['ptraj_sp'][event][0] + layer_dz*trajectories['pnorm_sp'][event][0], 
                trajectories['ptraj_sp'][event][1] + layer_dz*trajectories['pnorm_sp'][event][1], 
                trajectories['ptraj_sp'][event][2] + layer_dz*trajectories['pnorm_sp'][event][2])
    pLayerIntercepts.append(intercept)
    pLayerIntercepts = np.concatenate(np.array(pLayerIntercepts))
    
    hitsX = []
    hitsY = []
    hitsZ = []
    pdgids = []

    # loop through each hit within the event
    for hit in range(len(tree['EcalRecHits_v3_v12/EcalRecHits_v3_v12.xpos_'][event])):
        x = tree['EcalRecHits_v3_v12/EcalRecHits_v3_v12.xpos_'][event][hit]
        y = tree['EcalRecHits_v3_v12/EcalRecHits_v3_v12.ypos_'][event][hit]
        z = tree['EcalRecHits_v3_v12/EcalRecHits_v3_v12.zpos_'][event][hit]
        pdgid = tree['Pdg_ID'][event][hit]
        
        # loop through each layer of the ECal and check if its distance from the e- layer intercept is > corresponding layer's containment radius
        for layer in range(34):
            if z >= layer_z[layer] - .25 and z <= layer_z[layer] + .25:
                dist = np.sqrt((eLayerIntercepts[0][layer] - x)**2 + (eLayerIntercepts[1][layer] - y)**2)
                if dist >= radius_beam_68[layer]:
                    hitsX.append(x)
                    hitsY.append(y)
                    hitsZ.append(z)
                    pdgids.append(pdgid)

    eventsX.append(hitsX)
    eventsY.append(hitsY)
    eventsZ.append(hitsZ)
    pdgIDs.append(pdgids)

# these are our new branches without the hits inside the electron containment radius
cutTree['EcalRecHits_v3_v12/EcalRecHits_v3_v12.xpos_'] = eventsX
cutTree['EcalRecHits_v3_v12/EcalRecHits_v3_v12.ypos_'] = eventsY
cutTree['EcalRecHits_v3_v12/EcalRecHits_v3_v12.zpos_'] = eventsZ
cutTree['Pdg_ID'] = pdgIDs

#for i in range(len(tree['EcalRecHits_v3_v12/EcalRecHits_v3_v12.xpos_'])):
#    Hits1 = len(tree['EcalRecHits_v3_v12/EcalRecHits_v3_v12.xpos_'][i])
#    Hits2 = len(cutTree['EcalRecHits_v3_v12/EcalRecHits_v3_v12.xpos_'][i])
#    print('For event {}: Original Hits = {}, Outside Radii Hits = {}'.format(i, Hits1, Hits2))

import ROOT as r
from array import array 

file = r.TFile("skimmed_events_pdg.root","RECREATE")
file.cd()
tree = r.TTree("Events","Events")

hitX = r.std.vector('float')()
hitY = r.std.vector('float')()
hitZ = r.std.vector('float')()
pdgID = r.std.vector('int')()

# tree.Branch("hitN",hitN,"hitN/I")
tree.Branch("hitX",hitX)
tree.Branch("hitY",hitY)
tree.Branch("hitZ",hitZ)
tree.Branch("pdgID",pdgID)

for event in range(35):
    hitX.clear()
    hitY.clear()
    hitZ.clear()
    pdgID.clear()
    for hit in range(len(cutTree['EcalRecHits_v3_v12/EcalRecHits_v3_v12.xpos_'][event])):
        hitX.push_back(cutTree['EcalRecHits_v3_v12/EcalRecHits_v3_v12.xpos_'][event][hit])
        hitY.push_back(cutTree['EcalRecHits_v3_v12/EcalRecHits_v3_v12.ypos_'][event][hit])
        hitZ.push_back(cutTree['EcalRecHits_v3_v12/EcalRecHits_v3_v12.zpos_'][event][hit])
        pdgID.push_back(cutTree['Pdg_ID'][event][hit])

    tree.Fill()

file.cd()
tree.Write()
file.Close()