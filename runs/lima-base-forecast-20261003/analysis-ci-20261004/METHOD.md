# LIMA forecast plots with 95% confidence intervals

The error bars use the three training seeds as the independent units. Each seed has equal weight. Sampled answers are not additional training runs.

The forecast plot uses the median of all valid answers for each trained seed. Its central curve is the mean of these three medians. The base curve is the median of valid answers from the fixed pretrained reference. There is one base reference per model and format. It has no interval across training seeds.

The change plot first matches base and tuned answers by generation draw ID. For each seed and deadline, it takes the median of the paired probability differences. The central curve is the mean of these three seed medians. It can differ from the difference between the curves in the forecast plot. The original report used the median across seed medians. These new plots explicitly use the mean so that a Student t interval applies to the stated estimator.

For three seed estimates x, the interval is mean(x) ± t(0.975, 2) × sample_SD(x) / sqrt(3). The critical value is 4.3026527299. See [NIST: Confidence Limits for the Mean](https://www.itl.nist.gov/div898/handbook/eda/section3/eda352.htm).

These are pointwise 95% intervals. They are not a simultaneous band across deadlines, models, or formats. The calculation assumes independent, approximately normally distributed seed estimates. Three seeds cannot establish this assumption. The intervals describe variation from the training seed, conditional on the fixed prompts, generation seeds, base checkpoint, and valid outputs. They do not measure forecast accuracy, event uncertainty, or variation across independently pretrained checkpoints.

Invalid outputs can bias the reported forecasts and changes. They remain excluded under the original parser. The plots state valid counts for seeds 0, 1, and 2 in that order. Qwen3 ChatML has valid counts 1, 0, 1, with paired counts 1, 0, 1. Its complete three-seed estimate and interval are unavailable. Its two available seed curves are descriptive only. Missing estimates are not replaced with zero. Intervals are not clipped to the bounds of probabilities or changes. A zero-width interval means that the three observed seed medians are equal; it does not establish zero uncertainty.

The event is a correct complete solution to at least one of the six historically open Millennium problems, with an identifiable AI contribution incorporated into the solution. Partial progress and the Poincaré problem are excluded.

Reproduce from the campaign directory with `python3 plot_ci.py`. Dependencies: Python 3, SciPy, and Matplotlib. The script verifies all 720 raw forecast records and the training endpoints with the existing checks. It also verifies the original analysis hashes before and after plotting. The original analysis files remain intact.
