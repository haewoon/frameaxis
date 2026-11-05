# Corrected test vs the 2021 rule

Per aspect × sentiment cell: axes with p_2021 <= 0.05 (sign-chosen tail, N in the denominator), with two-sided p <= 0.05, and with BH q <= 0.05; then how many axes the 2021 rule kept that the corrected test drops, and the reverse.

## average

| cell | 2021 rule | two-sided p | BH q | dropped | added | top axis (2021 rule) | top axis (BH) |
|---|---|---|---|---|---|---|---|
| ambience_negative | 419 | 783 | 646 | 130 | 357 | ('savory', 'unsavory') | ('savory', 'unsavory') |
| ambience_positive | 819 | 1107 | 1066 | 70 | 317 | ('styleless', 'stylish') | ('styleless', 'stylish') |
| anecdotes-miscellaneous_negative | 840 | 1225 | 1197 | 57 | 414 | ('sparkling', 'still') | ('sparkling', 'still') |
| anecdotes-miscellaneous_positive | 1132 | 1292 | 1274 | 41 | 183 | ('sparkling', 'still') | ('sparkling', 'still') |
| food_negative | 281 | 1308 | 1296 | 14 | 1029 | ('impressive', 'unimpressive') | ('impressive', 'unimpressive') |
| food_positive | 524 | 1355 | 1344 | 24 | 844 | ('poor', 'rich') | ('poor', 'rich') |
| price_negative | 381 | 836 | 736 | 110 | 465 | ('ample', 'meager') | ('ample', 'meager') |
| price_positive | 779 | 971 | 901 | 121 | 243 | ('evil', 'good') | ('evil', 'good') |
| service_negative | 602 | 1151 | 1103 | 59 | 560 | ('best', 'worst') | ('best', 'worst') |
| service_positive | 968 | 1189 | 1164 | 59 | 255 | ('disrespectful', 'respectful') | ('disrespectful', 'respectful') |

## second_moment

| cell | 2021 rule | two-sided p | BH q | dropped | added | top axis (2021 rule) | top axis (BH) |
|---|---|---|---|---|---|---|---|
| ambience_negative | 213 | 549 | 140 | 130 | 57 | ('loud', 'soft') | ('loud', 'soft') |
| ambience_positive | 226 | 949 | 847 | 47 | 668 | ('active', 'quiet') | ('active', 'quiet') |
| anecdotes-miscellaneous_negative | 252 | 729 | 561 | 65 | 374 | ('best', 'worst') | ('best', 'worst') |
| anecdotes-miscellaneous_positive | 499 | 893 | 807 | 123 | 431 | ('go', 'no-go') | ('go', 'no-go') |
| food_negative | 730 | 765 | 671 | 128 | 69 | ('appealing', 'unappealing') | ('appealing', 'unappealing') |
| food_positive | 833 | 1061 | 1000 | 96 | 263 | ('savory', 'unsavory') | ('savory', 'unsavory') |
| price_negative | 237 | 391 | 167 | 122 | 52 | ('pointed', 'pointless') | ('pointed', 'pointless') |
| price_positive | 163 | 738 | 482 | 51 | 370 | ('cheap', 'expensive') | ('cheap', 'expensive') |
| service_negative | 430 | 582 | 378 | 142 | 90 | ('best', 'worst') | ('best', 'worst') |
| service_positive | 290 | 919 | 809 | 56 | 575 | ('hospitable', 'inhospitable') | ('hospitable', 'inhospitable') |

The top-axis columns follow notebook 4: the largest positive `diff_a_b`, which notebook 4 uses for the intensity figure. Before #13 it was taken over all axes; now only over q <= 0.05.
