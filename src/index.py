import pretty_errors
pretty_errors.configure(
    lines_before=3,
    display_locals=True,  # Enable the display of local variables
)

# import Experiments.MFP_2Player as TwoP
# import Experiments.MFP_3Player as ThreeP
# import Experiments.fairness as F
# import Experiments.MFPAgg_2Player as TwoAgg
import Experiments.MFP_4_to_6 as F2S 
# import Experiments.NNM as NN 
# import Experiments.MFPAgg_2_to_6 as Agg
# import Experiments.NNM as NNM
# import Experiments.random_model as R
# import Experiments.tests_rounds as T
# import Experiments.tests_simulations as TS
# import Experiments.parameter_fit as PF
# import Experiments.test_MFP_versions as Test
# import Experiments.test_latex_utils as LTX
# import Experiments.test_cogmods as CM
# import Experiments.test_parameter_fit_cog_models as PFCM
# import Reports.overview_wsls as WSLS
# import Reports.overview_PRW as PWR
# import Reports.overview_ARW as AWR
# import Reports.overview_Qlearning as QL
# import Reports.overview_Random as RAN
# import Reports.overview_MFP as MFP
# import Reports.overview_MFPAgg as MFPA
# import Reports.fit_models as FM
# import Experiments.ce_test as CE
# import Reports.model_analysis as MA
# import Experiments.test_optimizer as TE
# import Experiments.find_bug as BG
# import Figures.Fig1 as Fig1
# import Figures.Fig3 as Fig3
# import Experiments.tests_get_measures as TM


if __name__ == '__main__':
 
    print('Running...')
    # TS.sweep_MFP()
    # TS.sweep_MFPAgg()
    # TS.test_experiment_MPF()
    # R.random_simple_experiment()
    # R.random_multi_player_experiment()
    # R.random_two_player_maxlikelyhood()
    # R.random_multi_player_maxlikelyhood()
    # R.create_tests_data()
    # T.test_MFPAGG()
    # T.test_MFP()
    # FP2.draw_frequencies()
    # FP2.simple_draw_bar_attendances()
    # FP2.sweep_epsilon()
    # TwoP.simple_draw_bar_attendances()
    # TwoP.draw_suboptimal_attendances()
    # TwoP.compare_cooldown()
    # ThreeP.simple_draw_bar_attendances()
    # ThreeP.draw_optimal_attendances()
    # ThreeP.compare_cooldown()
    # F2S.very_simple_run()
    # F2S.simple_run()
    # F2S.sweep_belief_strength()
    # F2S.sweep_num_agents()
    # F2S.sweep_inverse_temperature()
    # F2S.find_best_parameters()
    F2S.simulate_best_parameters()
    # F2S.plot_two_variate_kdes()
    # F2S.draw_attendances()
    # F2S.sweep_epsilon()
    # F2S.sweep_belief_strength()
    # NN.draw_attendances()
    # F.draw_bar_attendances_2P()
    # F.draw_bar_attendances_3P()
    # TwoAgg.simple_draw_bar_attendances()
    # TwoAgg.simple_draw_scores()
    # Agg.draw_attendances()
    # NNM.simple_draw_bar_attendances()
    # PF.test_dev_random_p(0.1)
    # PF.test_dev_random()  
    # PF.test_parameter_fit()
    # Test.test()
    # CM.test_wsls()
    # CM.test_payoff_rescorla_wagner()
    # CM.test_attendance_rescorla_wagner()
    # CM.test_q_learning()
    # CM.test_MFP()
    # CM.test_MFP_Agg()
    # LTX.test_print_parameters()
    # WSLS.simple_run()
    # WSLS.sweep_drive_to_go()
    # PWR.examinations()
    # AWR.examinations()
    # RAN.examine()
    # MFP.examinations()
    # MFPA.examinations()
    # WSLS.full_report()
    # PWR.full_report()
    # AWR.full_report()
    # QL.full_report()
    # RAN.full_report()
    # MFP.full_report()
    # MFPA.full_report()
    # PF.test_parameter_fit()
    # PFCM.test_parameter_fit_Random()
    # PFCM.test_parameter_fit_WSLS()
    # PFCM.test_parameter_fit_PRW()
    # PFCM.test_parameter_fit_ARW()
    # PFCM.test_parameter_fit_Qlearning()
    # PFCM.test_parameter_fit_MFP()
    # PFCM.test_parameter_fit_MFPAgg()
    # FM.fit_models()
    # QL.explore_discount()
    # CE.test_ce()
    # MA.para_fig_random()
    # MA.para_fig_PRW_vs_ARW()
    # TE.test_get_measure()
    # TE.test_parameter_opt_random()
    # TE.test_parameter_opt_PRW()
    # BG.find_bug_random()
    # Fig1.top_panel()
    # Fig1.center_panel()
    # Fig3.top_panel()
    # Fig3.center_panels()
    # Fig3.bottom_panels()
    # TM.test_measures()
    # TM.test_kde()
    # TM.test_kde_per_player()
    # TM.test_kde_treatment()
    print('Done!')