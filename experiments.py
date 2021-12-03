"""
A set of experiments for testing the results reported by the paper in Table 5.

These models were all trained for 4 epochs on random masks
"""

paper_results = {
    "black-box (gold)": {"keep": .618, "delete": .622, "add": .102, "final": .447},
    "white-box (gold)": {"keep": .640, "delete": .570, "add": .114, "final": .441},
    "black-box (ir)": {"keep": .611, "delete": .543, "add": .194, "final": .419},
    "white-box (ir)": {"keep": .618, "delete": .590, "add": .144, "final": .452},
    "heuristic (ir)": {"keep": .652, "delete": .627, "add": .155, "final": .478},
    "masked lm": {"keep": .561, "delete": .529, "add": .078, "final": .389},
}


## lr 1.0x10^{-5} ##

# black-box (gold)
"""
PYTHONPATH=src python3 -m error_correction.corrector.run \
    --model_name_or_path t5-base \
    --output_dir /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/output/masker/model=t5-base,lr=1e-5,masker=random,mutation_source=false,mutation_target=false,labels=all/seed=42 \
    --do_predict \
    --train_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/blackbox_gold_test_genre_50_2.jsonl \
    --val_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/blackbox_gold_test_genre_50_2.jsonl \
    --test_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/blackbox_gold_test_genre_50_2.jsonl \
    --reader mask \
    --mutation_source true \
    --mutation_target false \
    --labels all
"""

# white-box (gold)
"""
PYTHONPATH=src python3 -m error_correction.corrector.run \
    --model_name_or_path t5-base \
    --output_dir /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/output/masker/model=t5-base,lr=1e-5,masker=random,mutation_source=false,mutation_target=false,labels=all/seed=42 \
    --do_predict \
    --train_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/whitebox_gold_test_genre_50_2.jsonl \
    --val_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/whitebox_gold_test_genre_50_2.jsonl \
    --test_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/whitebox_gold_test_genre_50_2.jsonl \
    --reader mask \
    --mutation_source true \
    --mutation_target false \
    --labels all

 'test_avg_gen_time': 0.016753193919505176,
 'test_avg_loss': 0.6366316080093384,
 'test_avg_sari_avgaddscore': 0.4478470975289812,
 'test_avg_sari_avgdelscore': 0.603731354287264,
 'test_avg_sari_avgkeepscore': 0.6109055606648136,
 'test_avg_sari_finalscore': 0.5541613374936863,
 'test_avg_summ_len': 28.84790760461977,
 'test_loss': tensor(0.6366, device='cuda:0'),
 'test_sari_avgaddscore': tensor(0.4478, device='cuda:0')}

"""
white_box_gold = {"keep": 0.6109055606648136, "delete": 0.603731354287264, "add": 0.4478470975289812, "final": 0.5541613374936863}

# black-box (ir)
"""
PYTHONPATH=src python3 -m error_correction.corrector.run \
    --model_name_or_path t5-base \
    --output_dir /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/output/masker/model=t5-base,lr=1e-5,masker=random,mutation_source=false,mutation_target=false,labels=all/seed=42 \
    --do_predict \
    --train_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/blackbox_ir_test_genre_50_2.jsonl \
    --val_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/blackbox_ir_test_genre_50_2.jsonl \
    --test_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/blackbox_ir_test_genre_50_2.jsonl \
    --reader mask \
    --mutation_source true \
    --mutation_target false \
    --labels all

 'test_avg_gen_time': 0.020429138643819778,
 'test_avg_loss': 1.2274504899978638,
 'test_avg_sari_avgaddscore': 0.5411005654871219,
 'test_avg_sari_avgdelscore': 0.6093276349908806,
 'test_avg_sari_avgkeepscore': 0.4380263618211851,
 'test_avg_sari_finalscore': 0.5294848540997291,
 'test_avg_summ_len': 32.601342628453395,
 'test_loss': tensor(1.2275, device='cuda:0'),
 'test_sari_avgaddscore': tensor(0.5411, device='cuda:0')}
"""
black_box_ir = {"keep": 0.4380263618211851, "delete": 0.6093276349908806, "add": 0.5411005654871219, "final": 0.5294848540997291}

# white-box (ir)
"""
PYTHONPATH=src python3 -m error_correction.corrector.run \
    --model_name_or_path t5-base \
    --output_dir /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/output/masker/model=t5-base,lr=1e-5,masker=random,mutation_source=false,mutation_target=false,labels=all/seed=42 \
    --do_predict \
    --train_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/whitebox_ir_test_genre_50_2.jsonl \
    --val_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/whitebox_ir_test_genre_50_2.jsonl \
    --test_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/whitebox_ir_test_genre_50_2.jsonl \
    --reader mask \
    --mutation_source true \
    --mutation_target false \
    --labels all

 'test_avg_gen_time': 0.01703997095588778,
 'test_avg_loss': 0.3337251543998718,
 'test_avg_sari_avgaddscore': 0.3325358437092247,
 'test_avg_sari_avgdelscore': 0.3893219053607588,
 'test_avg_sari_avgkeepscore': 0.7274597936187943,
 'test_avg_sari_finalscore': 0.48310584756292585,
 'test_avg_summ_len': 29.48428645028336,
 'test_loss': tensor(0.3337, device='cuda:0'),
 'test_sari_avgaddscore': tensor(0.3325, device='cuda:0')}
"""
white_box_ir = {"keep": 0.7274597936187943, "delete": 0.3893219053607588, "add": 0.3325358437092247, "final": 0.48310584756292585}

# heuristic (ir)
"""
PYTHONPATH=src python3 -m error_correction.corrector.run \
    --model_name_or_path t5-base \
    --output_dir /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/output/masker/model=t5-base,lr=1e-5,masker=random,mutation_source=false,mutation_target=false,labels=all/seed=42 \
    --do_predict \
    --train_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/heuristic_ir_test_genre_50_2.jsonl \
    --val_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/heuristic_ir_test_genre_50_2.jsonl \
    --test_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/heuristic_ir_test_genre_50_2.jsonl \
    --reader mask \
    --mutation_source true \
    --mutation_target false \
    --labels all

 'test_avg_gen_time': 0.01773841069798524,
 'test_avg_loss': 0.7846612930297852,
 'test_avg_sari_avgaddscore': 0.23433510184054035,
 'test_avg_sari_avgdelscore': 0.7383161704404735,
 'test_avg_sari_avgkeepscore': 0.7438958506088137,
 'test_avg_sari_finalscore': 0.5721823742966092,
 'test_avg_summ_len': 31.244298023314748,
 'test_loss': tensor(0.7847, device='cuda:0'),
 'test_sari_avgaddscore': tensor(0.2343, device='cuda:0')}
"""
heuristic_ir = {"keep": 0.7438958506088137, "delete": 0.7383161704404735, "add": 0.23433510184054035, "final": 0.5721823742966092}

# Percent difference for 1.0x10^{-5}

results = {
    "white-box (gold)": white_box_gold,
    "black-box (ir)": black_box_ir,
    "white-box (ir)": white_box_ir,
    "heuristic (ir)": heuristic_ir,
}

for result_key, value in results.items():
    paper_result = paper_results[result_key]
    for sari_key in value:
        percent_diff = ((value[sari_key] - paper_result[sari_key]) / paper_result[sari_key]) * 100
        print(f"{result_key}: {sari_key}: {percent_diff:.3f}")
    
    

## lr 5.0x10^{-5} ##

# black-box (gold)
"""
PYTHONPATH=src python3 -m error_correction.corrector.run \
    --model_name_or_path t5-base \
    --output_dir /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/output/masker/model=t5-base,lr=5e-5,masker=random,mutation_source=false,mutation_target=false,labels=all/seed=42 \
    --do_predict \
    --train_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/blackbox_gold_test_genre_50_2.jsonl \
    --val_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/blackbox_gold_test_genre_50_2.jsonl \
    --test_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/blackbox_gold_test_genre_50_2.jsonl \
    --reader mask \
    --mutation_source true \
    --mutation_target false \
    --labels all
"""

# white-box (gold)
"""
PYTHONPATH=src python3 -m error_correction.corrector.run \
    --model_name_or_path t5-base \
    --output_dir /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/output/masker/model=t5-base,lr=5e-5,masker=random,mutation_source=false,mutation_target=false,labels=all/seed=42 \
    --do_predict \
    --train_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/whitebox_gold_test_genre_50_2.jsonl \
    --val_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/whitebox_gold_test_genre_50_2.jsonl \
    --test_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/whitebox_gold_test_genre_50_2.jsonl \
    --reader mask \
    --mutation_source true \
    --mutation_target false \
    --labels all
 'test_avg_gen_time': 0.01702848721089805,
 'test_avg_loss': 0.9413727521896362,
 'test_avg_sari_avgaddscore': 0.3820510364889715,
 'test_avg_sari_avgdelscore': 0.5950886046065662,
 'test_avg_sari_avgkeepscore': 0.6086836295069809,
 'test_avg_sari_finalscore': 0.5286077568675062,
 'test_avg_summ_len': 30.51972401379931,
 'test_loss': tensor(0.9414, device='cuda:0'),
 'test_sari_avgaddscore': tensor(0.3821, device='cuda:0')}
"""
white_box_gold = {"keep": 0.6086836295069809, "delete": 0.5950886046065662, "add": 0.3820510364889715, "final": 0.5286077568675062}

# black-box (ir)
"""
PYTHONPATH=src python3 -m error_correction.corrector.run \
    --model_name_or_path t5-base \
    --output_dir /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/output/masker/model=t5-base,lr=5e-5,masker=random,mutation_source=false,mutation_target=false,labels=all/seed=42 \
    --do_predict \
    --train_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/blackbox_ir_test_genre_50_2.jsonl \
    --val_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/blackbox_ir_test_genre_50_2.jsonl \
    --test_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/blackbox_ir_test_genre_50_2.jsonl \
    --reader mask \
    --mutation_source true \
    --mutation_target false \
    --labels all

 'test_avg_gen_time': 0.020427323877811432,
 'test_avg_loss': 1.3414897918701172,
 'test_avg_sari_avgaddscore': 0.550371422516174,
 'test_avg_sari_avgdelscore': 0.6103341768193906,
 'test_avg_sari_avgkeepscore': 0.43920017390792127,
 'test_avg_sari_finalscore': 0.5333019244144953,
 'test_avg_summ_len': 32.799638523108705,
 'test_loss': tensor(1.3415, device='cuda:0'),
 'test_sari_avgaddscore': tensor(0.5504, device='cuda:0')}
"""
black_box_ir = {"keep": 0.43920017390792127, "delete": 0.6103341768193906, "add": 0.550371422516174, "final": 0.5333019244144953}

# white-box (ir)
"""
PYTHONPATH=src python3 -m error_correction.corrector.run \
    --model_name_or_path t5-base \
    --output_dir /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/output/masker/model=t5-base,lr=5e-5,masker=random,mutation_source=false,mutation_target=false,labels=all/seed=42 \
    --do_predict \
    --train_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/whitebox_ir_test_genre_50_2.jsonl \
    --val_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/whitebox_ir_test_genre_50_2.jsonl \
    --test_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/whitebox_ir_test_genre_50_2.jsonl \
    --reader mask \
    --mutation_source true \
    --mutation_target false \
    --labels all

 'test_avg_gen_time': 0.017166355814112993,
 'test_avg_loss': 0.39614346623420715,
 'test_avg_sari_avgaddscore': 0.3356417139929233,
 'test_avg_sari_avgdelscore': 0.389954836618207,
 'test_avg_sari_avgkeepscore': 0.7090156488719056,
 'test_avg_sari_finalscore': 0.47820406649434527,
 'test_avg_summ_len': 29.863472436888202,
 'test_loss': tensor(0.3961, device='cuda:0'),
 'test_sari_avgaddscore': tensor(0.3356, device='cuda:0')}

"""
white_box_ir = {"keep": 0.7090156488719056, "delete": 0.389954836618207, "add": 0.3356417139929233, "final": 0.47820406649434527}

# heuristic (ir)
"""
PYTHONPATH=src python3 -m error_correction.corrector.run \
    --model_name_or_path t5-base \
    --output_dir /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/output/masker/model=t5-base,lr=5e-5,masker=random,mutation_source=false,mutation_target=false,labels=all/seed=42 \
    --do_predict \
    --train_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/heuristic_ir_test_genre_50_2.jsonl \
    --val_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/heuristic_ir_test_genre_50_2.jsonl \
    --test_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/heuristic_ir_test_genre_50_2.jsonl \
    --reader mask \
    --mutation_source true \
    --mutation_target false \
    --labels all

 'test_avg_gen_time': 0.017936644252997017,
 'test_avg_loss': 0.7894928455352783,
 'test_avg_sari_avgaddscore': 0.2559714889082331,
 'test_avg_sari_avgdelscore': 0.7427541960676781,
 'test_avg_sari_avgkeepscore': 0.7447324249852806,
 'test_avg_sari_finalscore': 0.5811527033203973,
 'test_avg_summ_len': 31.88764994086839,
 'test_loss': tensor(0.7895, device='cuda:0'),
 'test_sari_avgaddscore': tensor(0.2560, device='cuda:0')}
"""
heuristic_ir = {"keep": 0.7447324249852806, "delete": 0.7427541960676781, "add": 0.255971488908233, "final": 0.5811527033203973}

# Percent difference for 1.0x10^{-5}

results = {
    "white-box (gold)": white_box_gold,
    "black-box (ir)": black_box_ir,
    "white-box (ir)": white_box_ir,
    "heuristic (ir)": heuristic_ir,
}
print("\n 5.0x10^{-5}", )
for result_key, value in results.items():
    paper_result = paper_results[result_key]
    for sari_key in value:
        percent_diff = ((value[sari_key] - paper_result[sari_key]) / paper_result[sari_key]) * 100
        print(f"{result_key}: {sari_key}: {percent_diff:.3f}")
    
    
