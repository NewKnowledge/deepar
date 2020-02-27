Tensorflow 2 implementation of Amazon DeepAR Time Series Forecasting algorithm (https://arxiv.org/abs/1704.04110).

Influenced by these two open-source implementations: https://github.com/arrigonialberto86/deepar and https://github.com/zhykoties/TimeSeries (pytorch).

**deepar/dataset**: 

1. **time_series.py**: contains *TimeSeriesTrain* and *TimeSeriesTest* objects that perform covariate augmentation, grouping, scaling, and standardization according to Salinas et al. The objects are also easy to integrate with the **D3M** AutoML DARPA primitive and piepline infrastructure (https://docs.datadrivendiscovery.org/).

**deepar/model**: 

1. **learner.py**: contains a *DeepARLearner* class, which creates the model structure and implements a custom training loop. The model learns a categorical embedding for each unique time series group. It also performs ancestral sampling during inference (for arbitrary horizons into the future) and generates *n* samples at each timestep. Ancestral sampling can be conditioned with the whole training time series or just the final window.

2. **layers.py**: contains custom *LSTMResetStateful* layer and *GaussianLayer* layer (the latter is from https://github.com/arrigonialberto86/deepar and unused in current codebase)

3. **loss.py**: contains custom *GaussianLogLikelihood* loss for real data and *NegativeBinomialLogLikelihood* loss for positive count data. Both losses support masking and inverse scaling per Salinas et al. 

<!-- ## TODO
    -DAR: 

        Qs:
            -Where do multiple embeddings live?
            -Constraints across multiple embedding spaces? (PIP?)
            -compare embeddings in same space to different space

        Reading
            -Optimal Rec -> statistically how does ARIMA + OR compare to deep ann + embeddings
                            is there a way to incorporate as constraint on embeddings (not forecasts)
            -Add Scientific Method steps (hypothesis, proposed experiments to doc) share w/ Craig and team for early feedback (while in process on reproduction)
            -ICML / KDD 2019 workshops (preference hierarchical, embeddings)

        Week 2 (1 package OSS, serializable)

            TRY TO REPRODUCE DeepAR (3-parts, electricity, traffic (want beta)) / Gluon experiments on public datasets w/ their splits (11 public datasets) 1-5 datasets (some grouped) + Australia in depth

            Share Scientific Method Doc: big pic = on the utility of unconstrained and constrained embeddings for forecasting grouped and hierarchical time series
            Hypothesis - 
                1) embeddings move forecast toward reconcilitation (why)
                2) soft OR w/ embedding aggregation constraint - different embedding spaces?)
            Finish OR Reading

            lr scheduling (10^-3 halve after 300 batches if no improvement or exponential)
                -test w/ tb
            generate multiple sample traces (use all samples as input at next state (higher default BS - new predictor network??)) - parallelism??, mask_value??
            multiple seasonality (with # back in time)
            student t instead of gaussian
            observed values indicator??

        Week 3 (rest OSS, serializable)
            Continue working toward reproduction
            Compare OR with Arima forecasts and OR with DeepAR forecasts w/ 0,1,2 embeddings (turn off in model) on Australia hierarchical example (with EDA)

        Week 4 (list to Craig / Ben, transfer rest of repos)
            Compare OR with Arima forecasts and OR with DeepAR forecasts w/ unconstrained, multiple embeddings
            Hypothesis: Improvement from OR technique on forecasts w/ embeddings = smaller
                -why: need to flesh this out more (should they without constraint??)
            
            soft OR-like aggregation constraint on embeddings -> brainstorm w/ Craig

        Week 5 (multivariate targets)
            soft OR-like aggregation constraint on embeddings -> brainstorm w/ Craig

        Week 6 (tests)
            soft OR-like aggregation constraint on embeddings -> brainstorm w/ Craig
            look at evolution of embeddings over training process

        Deliverable (Outline of Workshop Paper)
            -Related Work: Compile Citations + Notes as Reading
            -Reproduce DeepAR metrics: clear notebook w/ GluonTS + our impl.
            -Compare to OR: clear notebook w/ our impl. and clear R script (python impl??)
            -Methodology: comparison of embeddings to OR
            -Methodology: proposed constraint and why it makes sense
            -Results: Reproduction (?), comparison to OR, comparison to constraint      




    -->


    


