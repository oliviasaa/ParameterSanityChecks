""" This program is meant to test Mana generation and rewards"""
import numpy

decayFactors = [4290989755, 4287015898, 4283045721, 4279079221, 4275116394, 4271157237,
                4267201747, 4263249920, 4259301752, 4255357241, 4251416383,
                4247479175, 4243545613, 4239615693, 4235689414, 4231766770, 4227847759,
                4223932377, 4220020622, 4216112489, 4212207975, 4208307077,
                4204409792, 4200516116, 4196626046, 4192739579, 4188856710, 4184977438,
                4181101758, 4177229668, 4173361163, 4169496241, 4165634898,
                4161777132, 4157922938, 4154072313, 4150225254, 4146381758, 4142541822,
                4138705441, 4134872614, 4131043336, 4127217604, 4123395415,
                4119576766, 4115761654, 4111950074, 4108142024, 4104337501, 4100536502,
                4096739022, 4092945060, 4089154610, 4085367672, 4081584240,
                4077804312, 4074027884, 4070254954, 4066485518, 4062719573, 4058957115,
                4055198142, 4051442650, 4047690636, 4043942097, 4040197029,
                4036455429, 4032717295, 4028982622, 4025251408, 4021523650, 4017799344,
                4014078486, 4010361075, 4006647106, 4002936577, 3999229484,
                3995525824, 3991825594, 3988128791, 3984435412, 3980745453, 3977058911,
                3973375783, 3969696066, 3966019757, 3962346853, 3958677350,
                3955011245, 3951348535, 3947689218, 3944033289, 3940380746, 3936731586,
                3933085805, 3929443400, 3925804369, 3922168708, 3918536413,
                3914907483, 3911281913, 3907659701, 3904040843, 3900425337, 3896813179,
                3893204366, 3889598896, 3885996764, 3882397968, 3878802505,
                3875210372, 3871621566, 3868036083, 3864453920, 3860875075, 3857299544,
                3853727325, 3850158414, 3846592808, 3843030504, 3839471499,
                3835915790, 3832363374, 3828814248, 3825268408, 3821725853, 3818186578,
                3814650580, 3811117858, 3807588407, 3804062225, 3800539308,
                3797019654, 3793503259, 3789990121, 3786480237, 3782973602, 3779470216,
                3775970074, 3772473173, 3768979511, 3765489084, 3762001889,
                3758517924, 3755037186, 3751559671, 3748085377, 3744614300, 3741146437,
                3737681787, 3734220344, 3730762108, 3727307074, 3723855240,
                3720406602, 3716961158, 3713518905, 3710079840, 3706643960, 3703211262,
                3699781742, 3696355399, 3692932229, 3689512229, 3686095396,
                3682681728, 3679271221, 3675863872, 3672459679, 3669058639, 3665660748,
                3662266004, 3658874404, 3655485944, 3652100623, 3648718437,
                3645339383, 3641963459, 3638590661, 3635220986, 3631854432, 3628490996,
                3625130675, 3621773465, 3618419365, 3615068371, 3611720480,
                3608375690, 3605033997, 3601695399, 3598359893, 3595027476, 3591698145,
                3588371897, 3585048730, 3581728640, 3578411625, 3575097682,
                3571786808, 3568479000, 3565174255, 3561872571, 3558573944, 3555278373,
                3551985853, 3548696383, 3545409959, 3542126578, 3538846238,
                3535568936, 3532294669, 3529023435, 3525755230, 3522490051, 3519227897,
                3515968763, 3512712648, 3509459548, 3506209461, 3502962384,
                3499718314, 3496477248, 3493239183, 3490004118, 3486772048, 3483542972,
                3480316886, 3477093788, 3473873674, 3470656543, 3467442391,
                3464231216, 3461023014, 3457817784, 3454615522, 3451416225, 3448219892,
                3445026518, 3441836102, 3438648641, 3435464131, 3432282571,
                3429103957, 3425928286, 3422755557, 3419585766, 3416418910, 3413254987,
                3410093995, 3406935929, 3403780789, 3400628570, 3397479270,
                3394332887, 3391189418, 3388048860, 3384911211, 3381776467, 3378644627,
                3375515686, 3372389644, 3369266496, 3366146241, 3363028875,
                3359914396, 3356802802, 3353694089, 3350588256, 3347485298, 3344385214,
                3341288001, 3338193657, 3335102178, 3332013562, 3328927806,
                3325844909, 3322764866, 3319687675, 3316613335, 3313541841, 3310473192,
                3307407385, 3304344417, 3301284286, 3298226988, 3295172522,
                3292120885, 3289072074, 3286026086, 3282982919, 3279942570, 3276905037,
                3273870317, 3270838408, 3267809306, 3264783010, 3261759516,
                3258738822, 3255720926, 3252705824, 3249693515, 3246683996, 3243677263,
                3240673315, 3237672149, 3234673763, 3231678153, 3228685317,
                3225695253, 3222707958, 3219723430, 3216741666, 3213762662, 3210786418,
                3207812930, 3204842196, 3201874213, 3198908979, 3195946490,
                3192986746, 3190029742, 3187075477, 3184123947, 3181175151, 3178229086,
                3175285749, 3172345138, 3169407251, 3166472084, 3163539635,
                3160609902, 3157682882, 3154758573, 3151836972, 3148918077, 3146001885,
                3143088393, 3140177600, 3137269503, 3134364098, 3131461384,
                3128561359, 3125664019, 3122769362, 3119877387, 3116988089, 3114101467,
                3111217518, 3108336240, 3105457631, 3102581687, 3099708407,
                3096837788, 3093969827, 3091104522, 3088241871, 3085381870, 3082524519,
                3079669813, 3076817752, 3073968331, 3071121550, 3068277404,
                3065435893, 3062597013, 3059760763, 3056927139, 3054096139, 3051267761,
                3048442002, 3045618860, 3042798333, 3039980417, 3037165112,
                3034352413, 3031542320, 3028734829, 3025929938, 3023127644, 3020327946,
                3017530840, 3014736325, 3011944398, 3009155056]
decayFactorsLength = numpy.uint64(len(decayFactors))
decayFactorExp = numpy.uint64(32)
decayFactorEpochsSum = numpy.uint64(2262417561)
decayFactorEpochsSumExponent = numpy.uint64(21)
shimmerGenerationRate = numpy.uint64(1)
shimmerGenRateExponent = numpy.uint64(17) ## CHANGE THIS PROBABLY
#maxShimmerTokenSupply = numpy.uint64(1813620509061365)
maxShimmerTokenSupply = numpy.uint64(4600000000000000)
slotsPerEpochExponent = numpy.uint64(13)
annualdecayFactorPercentage = numpy.uint64(70)
bitsCount = numpy.uint64(63)
slotDurationInSeconds = numpy.uint64(10)
validationBlocksPerSlot = numpy.uint64(10)
#profitMarginExponent = numpy.uint64(13)
profitMarginExponent = numpy.uint64(8)
bootstrappingDuration = numpy.uint64(1079)
rewardToGenerationRatio = numpy.uint64(2)
#initialTargetRewardsRate = numpy.uint64(616067521149262)
#finalTargetRewardsRate = numpy.uint64(226702563632670)
initialTargetRewardsRate = numpy.uint64(1562570881354497)
finalTargetRewardsRate = numpy.uint64(575000000000000)
poolCoefficientExponent = numpy.uint64(11)
#poolCoefficientExponent = numpy.uint64(13)
maxToTargetRatio = numpy.uint64(10)

def print_a_line_of_dashes():
    """Prints some dashes"""
    print("-----------------------------------------------------------------------------------")

def sanity_check_lookup_table():
    """Sanity check for the lookup table"""
    seconds_in_an_epoch = slotDurationInSeconds << slotsPerEpochExponent
    seconds_in_a_year = 60 * 60 * 24* 365
    epoch_duration_in_years = seconds_in_an_epoch/seconds_in_a_year
    annual_dec_float = annualdecayFactorPercentage/100
    errors_count = 0
    print_a_line_of_dashes()
    print("Lookup table precision test: \n")
    for i, entry in enumerate(decayFactors):
        expected_float_result = annual_dec_float**((i+1)*epoch_duration_in_years)
        delta = expected_float_result*(2**decayFactorExp) - entry
        if delta >= 1 or delta < 0:
            print("FAILED lookup table precision test for entry", i)
            print("Expected // Actual", expected_float_result*(2**decayFactorExp), entry)
            errors_count = errors_count + 1
    if errors_count == 0:
        print("PASSED")
    else:
        print("Lookup table precision test ended; errors printed above")
    print_a_line_of_dashes()
    print("\n")

def sanity_check_initial_and_final_rewards():
    """Sanity check for the initial and final target rewards"""
    seconds_in_an_epoch = slotDurationInSeconds << slotsPerEpochExponent
    seconds_in_a_year = 60 * 60 * 24* 365
    epoch_duration_in_years = seconds_in_an_epoch/seconds_in_a_year
    bootstrapping_duration_in_years = epoch_duration_in_years * bootstrappingDuration
    annual_dec_float = annualdecayFactorPercentage/100
    final_rewards_float = maxShimmerTokenSupply * rewardToGenerationRatio * shimmerGenerationRate
    float_exp = numpy.float64(shimmerGenRateExponent)-numpy.float64(slotsPerEpochExponent)
    final_rewards_float = final_rewards_float * (2**(-float_exp))
    initial_rewards_float = final_rewards_float/(annual_dec_float**bootstrapping_duration_in_years)
    final_rewards_delta = final_rewards_float - finalTargetRewardsRate
    initial_rewards_delta = initial_rewards_float - initialTargetRewardsRate
    print_a_line_of_dashes()
    print("Initial and Final Rewards test:\n")
    if final_rewards_delta < 0 or final_rewards_delta >= 1:
        print("FAILED Final Reward sanity check (Expected // Actual)",
              final_rewards_float, finalTargetRewardsRate)
    else:
        print("Final Rewards: PASSED")
    if initial_rewards_delta < 0 or initial_rewards_delta >= 1:
        print("FAILED Initial Reward sanity check (Expected // Actual)",
              initial_rewards_float, initialTargetRewardsRate)
    else:
        print("Initial Rewards: PASSED")
    print_a_line_of_dashes()
    print("\n")

def sanity_check_mana_supply():
    """Sanity check for the total mana supply"""
    seconds_in_an_epoch = slotDurationInSeconds << slotsPerEpochExponent
    seconds_in_a_year = 60 * 60 * 24* 365
    epoch_duration_in_years = seconds_in_an_epoch/seconds_in_a_year
    annual_dec_float = annualdecayFactorPercentage/100
    epoch_decay = annual_dec_float**epoch_duration_in_years
    float_exp = numpy.float64(shimmerGenRateExponent)-numpy.float64(slotsPerEpochExponent)
    max_mana_gen_float=maxShimmerTokenSupply*shimmerGenerationRate*(2**(-float_exp))/(1-epoch_decay)
    max_mana_supply_float = max_mana_gen_float * (1+rewardToGenerationRatio*maxToTargetRatio)
    bits_used_for_mana_supply = numpy.floor(numpy.log2(max_mana_supply_float)) + 1
    print_a_line_of_dashes()
    print("Mana supply check:\n")

    if bits_used_for_mana_supply > bitsCount:
        print("FAILED mana supply check (supply bits // max bits))",
              bits_used_for_mana_supply, bitsCount)
    elif bits_used_for_mana_supply == bitsCount:
        print("PASSED mana supply check (supply // max))",
              bits_used_for_mana_supply, bitsCount)
    else:
        print("PASSED BUT NOT OPTIMAL mana supply check (supply // max))",
              bits_used_for_mana_supply, bitsCount)
        print("Consider increasing the generation rate.")

    print_a_line_of_dashes()
    print("\n")

def sanity_check_function_overflow():
    """Sanity check for the overflows in general"""
    print_a_line_of_dashes()
    print("General overflow tests:\n")
    errors_count = 0

    if maxShimmerTokenSupply*(2**profitMarginExponent) >= 2**64:
        print("FAILED overflow check 1")
        print("TokenSupply*(2**profitMarginExponent) >= 2**64")
        print("Actual value:", maxShimmerTokenSupply*(2**profitMarginExponent))
        print("2^64:", 2**64, "\n")
        errors_count = errors_count + 1

    if maxShimmerTokenSupply*(2**poolCoefficientExponent) >= 2**64:
        print("FAILED overflow check 2")
        print("TokenSupply*(2**poolCoefficientExponent) >= 2**64")
        print("Actual value:", maxShimmerTokenSupply*(2**poolCoefficientExponent))
        print("2^64:", 2**64, "\n")
        errors_count = errors_count + 1

    if initialTargetRewardsRate >= 2**(63-poolCoefficientExponent):
        print("FAILED overflow check 3")
        print("initialTargetRewardsRate => 2**(63-poolCoefficientExponent)")
        print("Actual value:", initialTargetRewardsRate)
        print("2**(63-poolCoefficientExponent):", 2**(63-poolCoefficientExponent), "\n")
        errors_count = errors_count + 1

    if initialTargetRewardsRate*validationBlocksPerSlot >= 2**(63):
        print("FAILED overflow check 4")
        print("initialTargetRewardsRate*validationBlocksPerSlot >= 2**63")
        print("Actual value:", initialTargetRewardsRate*validationBlocksPerSlot)
        print("2^63:", 2**63, "\n")
        errors_count = errors_count + 1

    if initialTargetRewardsRate >= 2**(64-profitMarginExponent):
        print("FAILED overflow check 5")
        print("initialTargetRewardsRate >= 2**(64-profitMarginExponent)")
        print("Actual value:", initialTargetRewardsRate)
        print("2**(64-profitMarginExponent):", 2**(64-profitMarginExponent), "\n")
        errors_count = errors_count + 1

    if slotsPerEpochExponent > shimmerGenRateExponent:
        print("FAILED overflow check 6")
        print("slotsPerEpochExponent > GenRateExponent")
        print("slotsPerEpochExponent:", slotsPerEpochExponent)
        print("GenRateExponent:", shimmerGenRateExponent, "\n")
        errors_count = errors_count + 1

    if errors_count == 0:
        print("PASSED all 6 tests")
    else:
        print("General overflow tests ended; errors printed above")
    print_a_line_of_dashes()
    print("\n")

def sanity_check_bootstrapping_duration():
    """Sanity check for the bootstrapping duration"""
    annual_dec_float = annualdecayFactorPercentage/100
    seconds_in_an_epoch = slotDurationInSeconds << slotsPerEpochExponent
    seconds_in_a_year = 60 * 60 * 24* 365
    epochs_in_an_year = seconds_in_a_year/seconds_in_an_epoch
    beta_per_year = -numpy.log(annual_dec_float)
    bootstrapping_duration_in_years = 1/beta_per_year
    bootstrapping_duration_in_epochs = numpy.uint64(
        bootstrapping_duration_in_years * epochs_in_an_year)
    print_a_line_of_dashes()
    print("Bootstrapping duration check: \n")

    if bootstrapping_duration_in_epochs != bootstrappingDuration:
        print("FAILED bootstrapping duration test (actual // expected)",
              bootstrappingDuration, bootstrapping_duration_in_epochs)
    else:
        print("PASSED")
    print_a_line_of_dashes()
    print("\n")

def sanity_check_decay_factor_epochs_sum():
    """Sanity check for the decay factor epoch sum"""
    annual_dec_float = annualdecayFactorPercentage/100
    seconds_in_an_epoch = slotDurationInSeconds << slotsPerEpochExponent
    seconds_in_a_year = 60 * 60 * 24* 365
    epoch_duration_in_years = seconds_in_an_epoch/seconds_in_a_year
    decay_per_epoch = annual_dec_float**epoch_duration_in_years
    dec_fac_ep_sum_float = decay_per_epoch/(1-decay_per_epoch)
    dec_fac_ep_sum_float = dec_fac_ep_sum_float*2**decayFactorEpochsSumExponent
    exp_dec_fac_ep_sum = numpy.uint64(dec_fac_ep_sum_float)
    print_a_line_of_dashes()
    print("Decay factor epoch sum check: \n")

    if exp_dec_fac_ep_sum != decayFactorEpochsSum:
        print("FAILED decay factor epoch sum test (actual // expected)",
              decayFactorEpochsSum, exp_dec_fac_ep_sum)
    else:
        print("PASSED")
    print_a_line_of_dashes()
    print("\n")

def additional_parameters_test():
    """Sanity check for other parameters"""
    print_a_line_of_dashes()
    print("Additional parameters checks: \n")
    errors_count = 0

    if not 0 <= decayFactorExp <= 32:
        print("Failed additional parameters test 1 (0 <= decayFactorExp <= 32)",
              decayFactorExp, "\n")
        errors_count = errors_count + 1

    if not 0 <= shimmerGenRateExponent <= 32:
        print("Failed additional parameters test 2 (0 <= GenRateExponent <= 32)",
              shimmerGenRateExponent, "\n")
        errors_count = errors_count + 1

    delta = decayFactorEpochsSumExponent+shimmerGenRateExponent-slotsPerEpochExponent
    if not 0 <= delta <= 32:
        print("Failed additional parameters test 3 (0 <= decayFactorEpochsSumExponent"
              +"+GenRateExponent-slotsPerEpochExponent <= 32)",
              delta, "\n")
        errors_count = errors_count + 1

    delta = shimmerGenRateExponent-slotsPerEpochExponent
    if not 0 <= delta <= 32:
        print("Failed additional param. test 4 (0 <= GenRateExponent-slotsPerEpochExponent <= 32)",
              delta, "\n")
        errors_count = errors_count + 1

    if decayFactorEpochsSum * shimmerGenerationRate >= 2**32:
        print("Failed additional parameters test 5 (decayFactorEpochsSum * GenerationRate < 2**32)",
              decayFactorEpochsSum * shimmerGenerationRate, 2**32, "\n")
        errors_count = errors_count + 1

    max_slot_index_diff = 2**32-1
    if max_slot_index_diff * shimmerGenerationRate >= 2**32:
        print("Failed additional parameters test 6 (max_slot_index_diff * GenerationRate < 2**32)",
              max_slot_index_diff * shimmerGenerationRate, 2**32, "\n")
        errors_count = errors_count + 1

    if errors_count == 0:
        print("PASSED all 6 tests")
    else:
        print("Additional parameters checks ended; errors printed above")

    print_a_line_of_dashes()
    print("\n")

sanity_check_lookup_table()
sanity_check_initial_and_final_rewards()
sanity_check_mana_supply()
sanity_check_function_overflow()
sanity_check_bootstrapping_duration()
sanity_check_decay_factor_epochs_sum()
additional_parameters_test()
