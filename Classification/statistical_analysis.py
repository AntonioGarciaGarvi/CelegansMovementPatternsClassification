from scipy import stats

f1_scores_freq_15_res_60 = [0.644, 0.605, 0.611]
f1_scores_freq_15_res_120 = [0.679, 0.762, 0.814]
f1_scores_freq_15_res_160 = [0.788, 0.822, 0.861]
f1_scores_freq_15_res_180 = [0.827, 0.854, 0.855]
f1_scores_freq_15_res_200 = [0.834, 0.838, 0.847]
f1_scores_freq_15_res_224 = [0.818, 0.845, 0.836]
f1_scores_freq_10_res_224 = [0.815, 0.814, 0.823]
f1_scores_freq_7_5_res_224 = [0.790, 0.847, 0.820]
f1_scores_freq_5_res_224 = [0.794, 0.846, 0.821]
f1_scores_freq_1_res_224 = [0.814, 0.806, 0.839]

f1_scores_lists = [f1_scores_freq_15_res_60, f1_scores_freq_15_res_120, f1_scores_freq_15_res_160,f1_scores_freq_15_res_180,f1_scores_freq_15_res_200, f1_scores_freq_15_res_224,
                   f1_scores_freq_10_res_224,f1_scores_freq_7_5_res_224, f1_scores_freq_5_res_224, f1_scores_freq_1_res_224]


alpha = 0.05 # significance level

print(f"-------------------------------------------------------")
print(f"Shapiro-Wilk test to check normality of the data")
print(f"-------------------------------------------------------")
# Perform the Shapiro-Wilk test for each list
for i, scores_list in enumerate(f1_scores_lists, 1):
    stat, p_value = stats.shapiro(scores_list)
    print(f"Shapiro-Wilk test for f1_scores_list_{i}: Statistic = {stat}, p-value = {p_value}")

    # Check if the p-value is less than 0.05 (common significance level)
    if p_value > alpha:
        print(f"f1_scores_list_{i} follows a normal distribution\n")
    else:
        print(f"f1_scores_list_{i} does not follow a normal distribution\n")


print(f"-------------------------------------------------------")
print(f"Bartlett's test to check equality of the variances")
print(f"-------------------------------------------------------")
# perform Bartlett's test for the frequency experiment
stat, p_value = stats.bartlett(f1_scores_freq_1_res_224, f1_scores_freq_5_res_224,f1_scores_freq_7_5_res_224, f1_scores_freq_10_res_224,f1_scores_freq_15_res_224)
print(f"frequency experiment Bartlett: stat = {stat}, p-value = {p_value}")

if p_value > alpha:
  print(f"the variances of different groups or samples are equal\n")
else:
  print(f"the variances of different groups or samples are different\n")

# perform Bartlett's test for the resolution experiment
stat, p_value = stats.bartlett(f1_scores_freq_15_res_60, f1_scores_freq_15_res_120, f1_scores_freq_15_res_160,f1_scores_freq_15_res_180,f1_scores_freq_15_res_200,f1_scores_freq_15_res_224)
print(f"resolution experiment Bartlett: stat = {stat}, p-value = {p_value}")
if p_value > alpha:
  print(f"the variances of different groups or samples are equal\n")
else:
  print(f"the variances of different groups or samples are different\n")

# perform ANOVA test for the resolution experiment
print(f"-------------------------------------------------------")
print(f"ANOVA test for the resolution experiment ")
print(f"-------------------------------------------------------")


f_val, p_val = stats.f_oneway(f1_scores_freq_15_res_60, f1_scores_freq_15_res_120, f1_scores_freq_15_res_160,f1_scores_freq_15_res_180,f1_scores_freq_15_res_200,f1_scores_freq_15_res_224)
print(f"f_oneway: stat = {f_val}, p-value = {p_val}")
if p_val > alpha:
    print("The means of the groups are not significantly different.\n")
else:
    print("The means of at least one group are significantly different.\n")

result = stats.tukey_hsd(f1_scores_freq_15_res_60, f1_scores_freq_15_res_120, f1_scores_freq_15_res_160,f1_scores_freq_15_res_180,
                   f1_scores_freq_15_res_200,f1_scores_freq_15_res_224)
print(result)

# perform ANOVA test for the frequency experiment
f_val, p_val = stats.f_oneway(f1_scores_freq_1_res_224, f1_scores_freq_5_res_224,f1_scores_freq_7_5_res_224, f1_scores_freq_10_res_224,f1_scores_freq_15_res_224)

print(f"-------------------------------------------------------")
print(f"ANOVA test for the frequency experiment ")
print(f"-------------------------------------------------------")
print(f"f_oneway: stat = {f_val}, p-value = {p_val}")
if p_val > alpha:
    print("The means of the groups are not significantly different.\n")
else:
    print("The means of at least one group are significantly different.\n")



print(f"\n-------------------------------------------------------")
print(f"Importance analysis of appearance information ")
print(f"-------------------------------------------------------")
f1_scores_freq_15_res_224 = [0.818, 0.845, 0.836]
f1_scores_freq_15_res_224Unet = [0.711, 0.721, 0.761]
f1_scores_freq_15_res_224APP = [0.837, 0.87, 0.835]
# # Check assumptions - Normality
# # # You can use Shapiro-Wilk test for normality
stat, p_value = stats.shapiro(f1_scores_freq_15_res_224)
print('f1_scores_freq_15_res_224')
print(f"Shapiro-Wilk test for normality: stat = {stat}, p-value = {p_value}")


stat, p_value = stats.shapiro(f1_scores_freq_15_res_224Unet)
print('f1_scores_freq_15_res_224Unet')
print(f"Shapiro-Wilk test for normality: stat = {stat}, p-value = {p_value}")


stat, p_value = stats.shapiro(f1_scores_freq_15_res_224APP)
print('f1_scores_freq_15_res_224APP')
print(f"Shapiro-Wilk test for normality: stat = {stat}, p-value = {p_value}")

# #
# Perform paired t-test
t_statistic, p_value = stats.ttest_rel(f1_scores_freq_15_res_224, f1_scores_freq_15_res_224Unet)

# Check if the results are statistically significant

if p_value < alpha:
    print("The differences between the two models are statistically significant (p = {}) using paired t-test".format(p_value))
else:
    print("The differences between the two models are not statistically significant (p = {}) using paired t-test".format(p_value))



t_statistic, p_value = stats.ttest_rel(f1_scores_freq_15_res_224, f1_scores_freq_15_res_224APP)

# Check if the results are statistically significant
if p_value < alpha:
    print("The differences between the two models are statistically significant (p = {}) using paired t-test".format(
        p_value))
else:
    print(
        "The differences between the two models are not statistically significant (p = {}) using paired t-test".format(
            p_value))
