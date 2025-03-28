
###############################
#
#    Coverage + Efficiency(Llama.cpp backend) Experiments 
#
#    Table 2 and Table 4 in the paper
#
# #############################


python run.py --engine guidance --config tests/configs/guidance.yaml --tasks Github_easy,Github_hard,Github_medium,Glaiveai2K,Kubernetes,Snowplow,WashingtonPost 


"""
+------------+-------------------+--------------------+------------+----------+-----------+---------+---------+
|    Task    | Declared coverage | Empirical coverage | Compliance | TTFT (s) | TPOT (ms) | TGT (s) | GCT (s) |
+------------+-------------------+--------------------+------------+----------+-----------+---------+---------+
| Glaiveai2K |        0.98       |        0.94        |    0.96    |   0.28   |    6.07   |   0.53  |   0.01  |
+------------+-------------------+--------------------+------------+----------+-----------+---------+---------+
| Github_easy |        0.85       |        0.82        |    0.96    |   0.35   |    7.09   |   0.61  |   0.01  |
+------------+-------------------+--------------------+------------+----------+-----------+---------+---------+
| WashingtonPost |        0.71       |        0.70        |    0.99    |   0.36   |    8.51   |   0.64  |   0.01  |
+----------------+-------------------+--------------------+------------+----------+-----------+---------+---------+
| Snowplow |        0.84       |        0.80        |    0.95    |   0.33   |    7.04   |   0.64  |   0.01  |
+---------------+-------------------+--------------------+------------+----------+-----------+---------+---------+
| Github_medium |        0.78       |        0.75        |    0.96    |   0.63   |    9.36   |   1.69  |   0.02  |
+------------+-------------------+--------------------+------------+----------+-----------+---------+---------+
| Kubernetes |        0.98       |        0.65        |    0.66    |   0.39   |    8.87   |   0.77  |   0.02  |
+------------+-------------------+--------------------+------------+----------+-----------+---------+---------+
| Github_hard |        0.61       |        0.40        |    0.66    |   1.26   |    8.43   |   2.02  |   0.07  |
+-------------+-------------------+--------------------+------------+----------+-----------+---------+---------+
"""




python run.py --engine outlines --config tests/configs/outlines.yaml --tasks Github_easy,Github_hard,Github_medium,Glaiveai2K,Kubernetes,Snowplow,WashingtonPost 

python run.py --engine llama_cpp --config tests/configs/llama_cpp.yaml --tasks Github_easy,Github_hard,Github_medium,Glaiveai2K,Kubernetes,Snowplow,WashingtonPost

python run.py --engine openai --config tests/configs/openai.yaml --tasks Github_easy,Github_hard,Github_medium,Glaiveai2K,Kubernetes,Snowplow,WashingtonPost

python run.py --engine gemini --config tests/configs/gemini.yaml --tasks Github_easy,Glaiveai2K

## Unconstrained

python run.py --engine llama_cpp --config tests/configs/llama_cpp_unconstrained.yaml --tasks Github_easy,Github_hard,Github_medium,Glaiveai2K,Kubernetes,Snowplow,WashingtonPost



###############################
#
#   Efficiency(Transformers backend) Experiments 
#
#   Table 3 in the paper
#
# #############################

python run.py --engine guidance --config tests/configs/guidance_tr.yaml --tasks Github_easy,Github_hard,Github_medium,Glaiveai2K,Kubernetes,Snowplow,WashingtonPost 

python run.py --engine xgrammar --config tests/configs/xgrammar.yaml --tasks Github_easy,Github_hard,Github_medium,Glaiveai2K,Kubernetes,Snowplow,WashingtonPost 

"""
+------------+-------------------+--------------------+------------+----------+-----------+---------+---------+
|    Task    | Declared coverage | Empirical coverage | Compliance | TTFT (s) | TPOT (ms) | TGT (s) | GCT (s) |
+------------+-------------------+--------------------+------------+----------+-----------+---------+---------+
| Glaiveai2K |        1.00       |        0.95        |    0.95    |   0.33   |   30.86   |   1.54  |   0.30  |
+-------------+-------------------+--------------------+------------+----------+-----------+---------+---------+
| Github_easy |        0.92       |        0.88        |    0.96    |   0.29   |   28.45   |   1.33  |   0.25  |
+----------------+-------------------+--------------------+------------+----------+-----------+---------+---------+
| WashingtonPost |        0.88       |        0.83        |     0.95    |   0.40   |   27.24   |   2.12  |   0.36  |
+----------+-------------------+--------------------+------------+----------+-----------+---------+---------+
| Snowplow |        0.47       |        0.42        |    0.89    |   0.39   |   31.38   |   0.12  |   0.35  |
+---------------+-------------------+--------------------+------------+----------+-----------+---------+---------+
| Github_medium |        0.87       |        0.69        |    0.79    |   0.47   |   25.87   |   3.72  |   0.43  |
+------------+-------------------+--------------------+------------+----------+-----------+---------+---------+
| Kubernetes |        0.12       |        0.12        |    1.00    |   1.42   |   32.45   |   0.11  |   1.31  |
+------------+-------------------+--------------------+------------+----------+-----------+---------+---------+
| Github_hard |        0.70       |        0.53        |    0.76    |   1.06   |   25.69   |   6.29  |   0.98  |
+-------------+-------------------+--------------------+------------+----------+-----------+---------+---------+
"""

python run.py --engine outlines --config tests/configs/outlines_tr.yaml --tasks Github_easy,Github_hard,Github_medium,Glaiveai2K,Kubernetes,Snowplow,WashingtonPost


## Unconstrained

python run.py --engine xgrammar --config tests/configs/xgrammar_unconstrained.yaml --tasks Github_easy,Github_hard,Github_medium,Glaiveai2K,Kubernetes,Snowplow,WashingtonPost


