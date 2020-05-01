### Always use capital letters for variables in a params file. 
import numpy as np

DATASETS = [
#     {'date':'20180108', 'name':'m52_2wk_post_tamox_real_1_test'}, ### THIS IS WHAT YOU TESTED CosmosDataset with
#    # {'date':'20180102', 'name':'m52_1wk_post_tamox_2'}
#    # {'date': '20180124', 'name': 'cux2ai148m72_vis_stim_2_test'},
#        {'date': '20180124', 'name': 'cux2ai148m72_vis_stim_2'}, ## Not that many cells recovered here. Redid the experiment.  
#     {'date': '20180212', 'name': 'm72_COSMOSShapeMultiGNG_1'} ### This was imported on 2/15/18. Looks great!
#      {'date': '20180213', 'name': 'm72_vis_stim_2'} ### This was imported on 2/15/18
#     {'date': '20180216', 'name': 'm73_COSMOSShapeMultiGNG_day2_1'},
#      {'date': '20180216', 'name': 'm72_COSMOSTrainMultiGNG_day2_1'}, 
#   {'date': '20180219', 'name':'cux2m72_COSMOSTrainMultiGNG_day5_1'},
##    {'date': '20180219', 'name': 'cux2m140_COSMOSShapeMultiGNG_1'},
##    {'date': '20180219', 'name': 'cux2m73_COSMOSTrainMultiGNG_day1_1'},
#     {'date': '20180221', 'name': 'cux2m140_vis_stim_2'},
#     {'date'ras: '20180221', 'name': 'cux2m140_vis_stim_3'},
#     {'date': '20180227', 'name': 'cux2m72_COSMOSTrainMultiBlockGNG_1'},
#       {'date': '20180227', 'name': 'testdata_cux2m72_COSMOSTrainMultiBlockGNG_1'}, ### test dataset (small).
#      {'date': '20180327', 'name': 'cux2m72_COSMOSTrainMultiBlockGNG_1'}, ### Imported this week of 3/30/18. pretty good behavior
    #{'date': '20180401', 'name': 'cux2ai148m72_COSMOSTrainMultiBlockGNG_1'}, ### Imported this 4/2/18. Very good behavior.
#     {'date': '20180404', 'name': 'cux2ai148m72_visual_stim_1'},
#     {'date': '20180404', 'name': 'cux2ai148m72_visual_stim_2_1'},
#     {'date': '20180404', 'name': 'cux2ai148m943_visual_stim_1'},
#       {'date': '20180419', 'name':'cux2ai148m72_visual_stim_vertical_1'},
#           {'date': '20180419', 'name':'cux2ai148m72_visual_stim_2'},
#   {'date': '20180420', 'name': 'cux2ai148m943_COSMOSTrainMultiBlockGNG_1',
#    'manual_keypoints': (np.array([[500, 650], [400, 650], [300, 650], [200, 600], [100, 540]]),
#                         np.array([[500, 180], [400, 190], [300, 195], [200, 210], [100, 240]]))},
#    {'date':'20180424', 'name': 'cux2ai148m945_cosmos_1'},
#    {'date':'20180424', 'name': 'cux2ai148m945_cosmos_2'},
#    {'date':'20180424', 'name': 'cux2ai148m945_cosmos_3'},
#    {'date': '20180424', 'name': 'cux2ai148m945_f1.2_1'},
#    {'date': '20180424', 'name': 'cux2ai148m945_f1.2_2'},
#    {'date': '20180424', 'name': 'cux2ai148m945_f1.2_3'},
#    {'date': '20180424', 'name': 'cux2ai148m945_f2_1'},
#    {'date': '20180424', 'name': 'cux2ai148m945_f2_2'},
#    {'date': '20180424', 'name': 'cux2ai148m945_f2_3'},
#    {'date': '20180424', 'name': 'cux2ai148m945_f2.8_1'},
#    {'date': '20180424', 'name': 'cux2ai148m945_f2.8_2'},
#    {'date': '20180424', 'name': 'cux2ai148m945_f2.8_3'},
#    {'date': '20180424', 'name': 'cux2ai148m945_f4_1'},
#    {'date': '20180424', 'name': 'cux2ai148m945_f4_2'},
#    {'date': '20180424', 'name': 'cux2ai148m945_f4_3'},
#    {'date': '20180424', 'name': 'cux2ai148m945_f5.6_1'},
#    {'date': '20180424', 'name': 'cux2ai148m945_f5.6_2'},
#    {'date': '20180424', 'name': 'cux2ai148m945_f5.6_3'},
#    {'date': '20180424', 'name': 'cux2ai148m945_f8_1'},
#    {'date': '20180424', 'name': 'cux2ai148m945_f8_2'},
#    {'date': '20180424', 'name': 'cux2ai148m945_f8_3'},
#    {'date': '20180424', 'name': 'cux2ai148m945_f1.2_4'},
#    {'date': '20180424', 'name': 'cux2ai148m945_cosmos_4'},

#      {'date': '20180523', 'name': 'cux2ai148m192_vis_stim_1'},
#      {'date': '20180523', 'name': 'cux2ai148m945_vis_stim_1'},
#      {'date': '20180522', 'name': 'cux2ai148m943_visual_stim_1'},
#      {'date': '20180522', 'name': 'cux2ai148m194_visual_stim_1'},
     # {'date': '20180522', 'name': 'cux2ai148m943_visual_stim_2'},
     # {'date': '20180522', 'name': 'cux2ai148m194_visual_stim_2'},
     # {'date': '20180522', 'name': 'cux2ai148m945_visual_stim_3'},
    
#     {'date': '20180520', 'name': 'cux2ai148m194_COSMOSTrainMultiBlockGNG_1',
#      'manual_keypoints': (np.array([[500, 650], [400, 650], [300, 650], [200, 650], [100, 650]]),
#                           np.array([[500, 150], [400, 150], [300, 175], [200, 180], [100, 200]]))},

#       {'date': '20180514', 'name': 'cux2ai148m72_COSMOSTrainMultiBlockGNG_1'},
#       {'date': '20180227', 'name': 'testdata_cux2m72_COSMOSTrainMultiBlockGNG_1'},
     # {'date': '20180626', 'name': 'cux2ai148m945_COSMOSTrainMultiBlockGNG_1'},
      # {'date':'20180709', 'name':'cux2ai148m194_COSMOSTrainMultiBlockGNG_2'}
#        {'date': '20180707', 'name': 'cux2ai148m194_COSMOSTrainMultiBlockGNG_1'},  # - Fine, no motion. Blue light turned off at end, be careful. -- imported
#    {'date': '20180514', 'name':'cux2ai148m72_COSMOSTrainMultiBlockGNG_2'}, # + Near perfect accuracy, window maybe not as good? # Top no movement really. same with bottom. looks great -- imported

#    {'date': '20180511', 'name':'cux2ai148m72_COSMOSTrainMultiBlockGNG_1'}, # + 85% # Top no movement really, great. Bot seems to have more movement. On the edge of usable. ----------- couldnt import
#    {'date': '20180430', 'name': 'cux2ai148m943_COSMOSTrainMultiBlockGNG_1'}, # + 84% # Top has some jiggle at end, bot has no movement -- imported (692 rois)
#    {'date': '20180424', 'name': 'cux2ai148m943_COSMOSTrainMultiBlockGNG_1'}, # + peak 94% # No movement, one jiggle in middle of top but ok, bot has a little more movement -- imported (829 rois) looks ok
#    {'date': '20180629', 'name':'cux2ai148m945_COSMOSTrainMultiBlockGNG_1'} # + Top has some jumps (maybe 2-3 pixels radius, seems like something was twisting?). Same in bottom... -- imported 320 rois, looks crappy
    
    ## Note: manual_keypoints for one of the 5 datasets above (not sure which though):
    ## manual_keypoints = (np.array([[500, 650], [400, 650], [300, 650], [200, 600], [100, 540]]) - [0, 150],
    ##                np.array([[500, 180], [400, 190], [300, 195], [200, 210], [100, 240]]) + [0, 100])
    
#    {'date': '20180624', 'name':'cux2ai148m945_COSMOSTrainMultiBlockGNG_1',
#     'manual_keypoints': (np.array([[500, 570], [400, 550], [300, 540], [200, 490], [100, 450]]),
#                         np.array([[500, 180], [400, 190], [300, 195], [200, 220], [100, 260]]))}, # peak 81% # Top essentially no movement. Bot also good. ---> Imported
    
  
#   {'date':'20180630', 'name':'cux2ai148m945_COSMOSTrainMultiBlockGNG_1', 
#    'manual_keypoints':(np.array([[500, 600], [400, 590], [300, 560], [200, 520], [100, 480]]),
#                       np.array([[500, 220], [400, 225], [300, 240], [200, 260], [100, 280]]))}, # 90% top no movement, bot no movement. Growth on the right side. But workable. 
###   {'date': '20180724', 'name': 'cux2ai148m194_COSMOSTrainMultiBlockGNG_1'}, # 86% top no movement. bot no movement. note weird thing in visual cortex. Not focused great, a quarter of the brain cut off (right motor cortex).
#     {'date': '20180723', 'name': 'cux2ai148m194_COSMOSTrainMultiBlockGNG_1'}, # 83% top minimal movement, bot minimal movement. Same issue with motor cortex on the right side. 

    
    # to import
    #0630 945 (90%)
    #0628 945 (76%)
    #0724 194 (86%)
    #0723 194 (83%)
    #0722 194 (79.2%) (where is this data?)
    #0721 194 (79.6%) (where is this data?)

#    {'date': '20180828', 'name': 'cux2ai148m4116_COSMOSTrainMultiBlockGNG_1',
#     'manual_keypoints': (np.array([[500, 600], [400, 590], [300, 550], [200, 450], [100, 420]]),
#                         np.array([[500, 220], [400, 225], [300, 250], [200, 310], [100, 340]]))}, # 83%. Some movement but it has been restricted to under a pixel radius.
    
#     {'date': '20180831', 'name':'cux2ai148m4116_COSMOSTrainMultiBlockGNG_1',
#      'manual_keypoints': (np.array([[500, 600], [400, 590], [300, 550], [200, 450], [100, 420]]),
#                     np.array([[500, 220], [400, 225], [300, 250], [200, 310], [100, 340]]))},

    
### START IMPORTING BELOW####
#         {'date': '20180205', 'name':'COSMOSShapeMultiGNG-m72_1'}, # movie1 is justOdors 10k frames. No movement,good.
#        {'date': '20180214', 'name':'m140_initial_odor_exposure__1', 
#         'manual_keypoints': (np.array([[500, 450], [400, 430], [300, 410], [200, 400], [100, 350]]), #[topy, topx]
#                              np.array([[500, 100], [400, 110], [300, 130], [200, 150], [100, 200]]))}, # justOdor, ~10k frames. No movement, good.
#      {'date': '20180328', 'name':'cux2ai148m943_COSMOSShapeMultiGNG_justOdor_1',
#       'manual_keypoints': (np.array([[500, 470], [400, 460], [300, 440], [200, 420], [100, 380]]), #[topy, topx]
#                            np.array([[500, 220], [400, 230], [300, 250], [200, 250], [100, 280]]))}, #justOdor. Looks good. 
#      {'date': '20180328', 'name':'cux2ai148m945_COSMOSShapeMultiGNG_justOdor_1', 
#       'manual_keypoints': (np.array([[500, 470], [400, 460], [300, 440], [200, 420], [100, 380]]), #[topy, topx]
#                             np.array([[500, 220], [400, 230], [300, 250], [200, 250], [100, 280]]))},
#     {'date': '20180328', 'name': 'cux2ai148m65_COSMOSShapeMultiGNG_justOdor_1',
#      'manual_keypoints': (np.array([[500, 500], [400, 490], [300, 470], [200, 450], [100, 380]]), #[topy, topx]
#                            np.array([[500, 150], [400, 170], [300, 180], [200, 190], [100, 230]]))},  # justOdor  Looks great. LED successfully removed.  ### 20190724This guy had weird bright expression/seizure
#      {'date': '20180205', 'name': 'COSMOSShapeMultiGNG-m72_5', 
#       'manual_keypoints': (np.array([[500, 470], [400, 460], [300, 450], [200, 420], [100, 350]]), #[topy, topx]
#                            np.array([[500, 180], [400, 200], [300, 210], [200, 220], [100, 260]]))}, # odors + spout, autoprob=1.  Then 2-4  COSMOSShapeMultiGNG, maybe movie 5 actually. Good.
#      {'date': '20180219', 'name': 'cux2m140_COSMOSShapeMultiGNG_1',
#       'manual_keypoints': (np.array([[500, 490], [400, 480], [300, 470], [200, 440], [100, 370]]), #[topy, topx]
#                            np.array([[500, 100], [400, 120], [300, 130], [200, 140], [100, 180]]))},  # odors + spout, autoprob=1. good.     {'date': '20180328', 'name':'cux2ai148m945_COSMOSShapeMultiGNG_justOdor_1', 
#       'manual_keypoints': (np.array([[500, 470], [400, 460], [300, 440], [200, 420], [100, 380]]), #[topy, topx]
#                             np.array([[500, 220], [400, 230], [300, 250], [200, 250], [100, 280]]))},
#      {'date': '20180328', 'name': 'cux2ai148m943_COSMOSShapeMultiGNG_1',
#       'manual_keypoints': (np.array([[500, 490], [400, 480], [300, 430], [200, 400], [100, 350]]), #[topy, topx]
#                            np.array([[500, 200], [400, 220], [300, 230], [200, 240], [100, 280]]))},  # odors + spout, autoprob=1. Looks good.
#      {'date': '20180328', 'na20190724me': 'cux2ai148m945_COSMOSShapeMultiGNG_1',
#       'manual_keypoints': (np.array([[500, 490], [400, 480], [300, 430], [200, 400], [100, 350]]), #[topy, topx]
#                            np.array([[500, 230], [400, 250], [300, 260], [200, 270], [100, 310]]))},  # odors + spout, autoprob=1. Looks good.
    #{'date' : '20190604', 'name' : 'cux2m4293_oddball_merged_orig_14_then_reversed_14'}

#     {'date': '20181125', 'name': 'thy1gcm1_COSMOSTrainMultiBlockGNG_1'},
#    {'date': '20190522', 'name': 'cux2m4293_lsd_1',
#     'manual_keypoints' : (np.array([[500, 550], [400, 540], [300, 520], [200, 460], [100, 400]]), #[topy, topx]
#                    np.array([[500, 110], [400, 120], [300, 140], [200, 170], [100, 200]]))},
#     {'date' : '20190502', 'name' : 'm4293_1hr_resting_state_3', 
#     'manual_keypoints' : (np.array([[500, 530], [400, 520], [300, 505], [200, 445], [100, 400]]), #[topy, topx]
#                     np.array([[500, 100], [400, 120], [300, 150], [200, 170], [100, 220]]))}
#   {'date': '20190604', 'name': 'cux2m4293_oddball_merged_orig_14_then_reversed_14', 
#     'manual_keypoints': (np.array([[500, 530], [400, 520], [300, 505], [200, 445], [100, 400]]), #[topy, topx]
#                          np.array([[500, 100], [400, 120], [300, 150], [200, 170], [100, 220]]))},  ### Trimmed in imagej. looks good. Exactly half of the frames are from first session.
#     {'date' : '20190521', 'name' : 'cux2m4293_saline_day2_1', 
#     'manual_keypoints' : (np.array([[500, 480], [400, 475], [300, 455], [200, 420], [100, 415]]), #[topy, topx]
#                           np.array([[500, 90], [400, 110], [300, 145], [200, 165], [100, 220]]))
#     } # culled by JK
    #{'date': '20190604', 'name': 'cux2m4293_saline_round3_1'}
   # {'date': '20190805', 'name': 'm4293_vis_stim_1'}
#     {'date' : '20190924', 'name' : 'VIP_m6534_test1_2'}
#    {'date' : '20190604', 'name' : 'cux2m4293_oddball_merged_orig_14_then_reversed_14'}
      {'date':'20180328', 'name':'cux2ai148m943_for_testing_cosmostools3'}
]

#TODO: 
#{'date' : '20190604', 'name' : 'cux2m4293_oddball_merged_orig_14_then_reversed_14'}
# 20190604 cux2m4293_saline_round3_1
# 20190604 cux2m5456_saline_1
# 20190605 cux2m5456_lsd_1
# 20190605 cux2m4293_lsd_round2_1
# 20190520 cux2m4293_saline_1 <-- do these data exist??
# Still need to cull 20190604 cux2m4293_oddball_merged_orig_14_then_reversed_14!!
# Need to run CNMFE on:
#     {'date' : '20190605', 'name' : 'cux2m5456_lsd_1'}
#     {'date' : '20190605', 'name' : 'cux2m4293_lsd_round2_1'}



