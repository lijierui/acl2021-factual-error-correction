"""
A set of experiments for testing the results reported by the paper in Table 5.

These models were all trained for 4 epochs on random masks
"""

paper_results = {
    "black-box (gold)": {"keep": 0.618, "delete": 0.622, "add": 0.102, "final": 0.447},
    "white-box (gold)": {"keep": 0.640, "delete": 0.570, "add": 0.114, "final": 0.441},
    "black-box (ir)": {"keep": 0.611, "delete": 0.543, "add": 0.194, "final": 0.419},
    "white-box (ir)": {"keep": 0.618, "delete": 0.590, "add": 0.144, "final": 0.452},
    "heuristic (ir)": {"keep": 0.652, "delete": 0.627, "add": 0.155, "final": 0.478},
    "masked lm": {"keep": 0.561, "delete": 0.529, "add": 0.078, "final": 0.389},
}


## lr 1.0x10^{-5} ##

# black-box (gold)
"""
PYTHONPATH=src python3 -m error_correction.corrector.run \
    --model_name_or_path t5-base \
    --output_dir /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/output/masker/model=t5-base,lr=1e-5,masker=random,mutation_source=false,mutation_target=false,labels=all/seed=42 \
    --do_predict \
    --train_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/blackbox_gold_test_genre_50_2_new.jsonl \
    --val_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/blackbox_gold_test_genre_50_2_new.jsonl \
    --test_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/blackbox_gold_test_genre_50_2_new.jsonl \
    --reader mask \
    --mutation_source true \
    --mutation_target false \
    --labels all

 'test_avg_loss': 0.3292052745819092,
 'test_avg_sari_avgaddscore': 0.32634621813816955,
 'test_avg_sari_avgdelscore': 0.4404458480548191,
 'test_avg_sari_avgkeepscore': 0.6780864178866625,
 'test_avg_sari_finalscore': 0.48162616135988373,
 'test_avg_summ_len': 30.546713971954723,
 'test_loss': tensor(0.3292, device='cuda:0'),
 'test_sari_avgaddscore': tensor(0.3263, device='cuda:0')}
"""
black_box_gold = {
    "keep": 0.6780864178866625,
    "delete": 0.4404458480548191,
    "add": 0.32634621813816955,
    "final": 0.48162616135988373,
}

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
white_box_gold = {
    "keep": 0.6109055606648136,
    "delete": 0.603731354287264,
    "add": 0.4478470975289812,
    "final": 0.5541613374936863,
}

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
black_box_ir = {
    "keep": 0.4380263618211851,
    "delete": 0.6093276349908806,
    "add": 0.5411005654871219,
    "final": 0.5294848540997291,
}

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
white_box_ir = {
    "keep": 0.7274597936187943,
    "delete": 0.3893219053607588,
    "add": 0.3325358437092247,
    "final": 0.48310584756292585,
}

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
heuristic_ir = {
    "keep": 0.7438958506088137,
    "delete": 0.7383161704404735,
    "add": 0.23433510184054035,
    "final": 0.5721823742966092,
}

# Percent difference for 1.0x10^{-5}

results = {
    "black-box (gold)": black_box_gold,
    "white-box (gold)": white_box_gold,
    "black-box (ir)": black_box_ir,
    "white-box (ir)": white_box_ir,
    "heuristic (ir)": heuristic_ir,
}
print(
    "\n 1.0x10^{-5}",
)
for result_key, value in results.items():
    paper_result = paper_results[result_key]
    for sari_key in value:
        percent_diff = (
            (value[sari_key] - paper_result[sari_key]) / paper_result[sari_key]
        ) * 100
        print(f"{result_key}: {sari_key}: {percent_diff:.3f}")


## lr 5.0x10^{-5} ##

# black-box (gold)
"""
PYTHONPATH=src python3 -m error_correction.corrector.run \
    --model_name_or_path t5-base \
    --output_dir /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/output/masker/model=t5-base,lr=5e-5,masker=random,mutation_source=false,mutation_target=false,labels=all/seed=42 \
    --do_predict \
    --train_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/blackbox_gold_test_genre_50_2_new.jsonl \
    --val_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/blackbox_gold_test_genre_50_2_new.jsonl \
    --test_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/blackbox_gold_test_genre_50_2_new.jsonl \
    --reader mask \
    --mutation_source true \
    --mutation_target false \
    --labels all

 'test_avg_loss': 0.34338510036468506,
 'test_avg_sari_avgaddscore': 0.3402580062296789,
 'test_avg_sari_avgdelscore': 0.4408150359430137,
 'test_avg_sari_avgkeepscore': 0.664862474177716,
 'test_avg_sari_finalscore': 0.48197850545013615,
 'test_avg_summ_len': 30.287210677479305,
 'test_loss': tensor(0.3434, device='cuda:0'),
 'test_sari_avgaddscore': tensor(0.3403, device='cuda:0')}

"""
black_box_gold = {
    "keep": 0.664862474177716,
    "delete": 0.4408150359430137,
    "add": 0.3402580062296789,
    "final": 0.48197850545013615,
}


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
white_box_gold = {
    "keep": 0.6086836295069809,
    "delete": 0.5950886046065662,
    "add": 0.3820510364889715,
    "final": 0.5286077568675062,
}

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
black_box_ir = {
    "keep": 0.43920017390792127,
    "delete": 0.6103341768193906,
    "add": 0.550371422516174,
    "final": 0.5333019244144953,
}

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
white_box_ir = {
    "keep": 0.7090156488719056,
    "delete": 0.389954836618207,
    "add": 0.3356417139929233,
    "final": 0.47820406649434527,
}

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
heuristic_ir = {
    "keep": 0.7447324249852806,
    "delete": 0.7427541960676781,
    "add": 0.255971488908233,
    "final": 0.5811527033203973,
}

# Percent difference for 1.0x10^{-5}

results = {
    "black-box (gold)": black_box_gold,
    "white-box (gold)": white_box_gold,
    "black-box (ir)": black_box_ir,
    "white-box (ir)": white_box_ir,
    "heuristic (ir)": heuristic_ir,
}
print(
    "\n 5.0x10^{-5}",
)
for result_key, value in results.items():
    paper_result = paper_results[result_key]
    for sari_key in value:
        percent_diff = (
            (value[sari_key] - paper_result[sari_key]) / paper_result[sari_key]
        ) * 100
        print(f"{result_key}: {sari_key}: {percent_diff:.3f}")


## lr 1.0x10^{-4} ##

# black-box (gold)
"""
PYTHONPATH=src python3 -m error_correction.corrector.run \
    --model_name_or_path t5-base \
    --output_dir /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/output/masker/model=t5-base,lr=1e-4,masker=random,mutation_source=false,mutation_target=false,labels=all/seed=42 \
    --do_predict \
    --train_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/blackbox_gold_test_genre_50_2_new.jsonl \
    --val_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/blackbox_gold_test_genre_50_2_new.jsonl \
    --test_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/blackbox_gold_test_genre_50_2_new.jsonl \
    --reader mask \
    --mutation_source true \
    --mutation_target false \
    --labels all

 'test_avg_loss': 0.38195183873176575,
 'test_avg_sari_avgaddscore': 0.33852361394670644,
 'test_avg_sari_avgdelscore': 0.43998402493397426,
 'test_avg_sari_avgkeepscore': 0.6711288516039153,
 'test_avg_sari_finalscore': 0.48321216349486534,
 'test_avg_summ_len': 30.74674776144619,
 'test_loss': tensor(0.3820, device='cuda:0'),
 'test_sari_avgaddscore': tensor(0.3385, device='cuda:0')}

"""
black_box_gold = {
    "keep": 0.6711288516039153,
    "delete": 0.43998402493397426,
    "add": 0.33852361394670644,
    "final": 0.48321216349486534,
}

# white-box (gold)
"""
PYTHONPATH=src python3 -m error_correction.corrector.run \
    --model_name_or_path t5-base \
    --output_dir /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/output/masker/model=t5-base,lr=1e-4,masker=random,mutation_source=false,mutation_target=false,labels=all/seed=42 \
    --do_predict \
    --train_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/whitebox_gold_test_genre_50_2.jsonl \
    --val_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/whitebox_gold_test_genre_50_2.jsonl \
    --test_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/whitebox_gold_test_genre_50_2.jsonl \
    --reader mask \
    --mutation_source true \
    --mutation_target false \
    --labels all

 'test_avg_gen_time': 0.01714222295523312,
 'test_avg_loss': 1.2314934730529785,
 'test_avg_sari_avgaddscore': 0.35790652945787654,
 'test_avg_sari_avgdelscore': 0.5895004465144708,
 'test_avg_sari_avgkeepscore': 0.607226974217489,
 'test_avg_sari_finalscore': 0.5182113167299455,
 'test_avg_summ_len': 31.484625768711563,
 'test_loss': tensor(1.2315, device='cuda:0'),
 'test_sari_avgaddscore': tensor(0.3579, device='cuda:0')}
"""
white_box_gold = {
    "keep": 0.607226974217489,
    "delete": 0.5895004465144708,
    "add": 0.35790652945787654,
    "final": 0.5182113167299455,
}

# black-box (ir)
"""
PYTHONPATH=src python3 -m error_correction.corrector.run \
    --model_name_or_path t5-base \
    --output_dir /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/output/masker/model=t5-base,lr=1e-4,masker=random,mutation_source=false,mutation_target=false,labels=all/seed=42 \
    --do_predict \
    --train_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/blackbox_ir_test_genre_50_2.jsonl \
    --val_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/blackbox_ir_test_genre_50_2.jsonl \
    --test_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/blackbox_ir_test_genre_50_2.jsonl \
    --reader mask \
    --mutation_source true \
    --mutation_target false \
    --labels all

 'test_avg_gen_time': 0.020474968569689108,
 'test_avg_loss': 1.486007809638977,
 'test_avg_sari_avgaddscore': 0.5383879291417792,
 'test_avg_sari_avgdelscore': 0.6096768270794498,
 'test_avg_sari_avgkeepscore': 0.4385013686847314,
 'test_avg_sari_finalscore': 0.5288553749686534,
 'test_avg_summ_len': 33.96462690420862,
 'test_loss': tensor(1.4860, device='cuda:0'),
 'test_sari_avgaddscore': tensor(0.5384, device='cuda:0')}


"""
black_box_ir = {
    "keep": 0.4385013686847314,
    "delete": 0.6096768270794498,
    "add": 0.5383879291417792,
    "final": 0.5288553749686534,
}

# white-box (ir)
"""
PYTHONPATH=src python3 -m error_correction.corrector.run \
    --model_name_or_path t5-base \
    --output_dir /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/output/masker/model=t5-base,lr=1e-4,masker=random,mutation_source=false,mutation_target=false,labels=all/seed=42 \
    --do_predict \
    --train_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/whitebox_ir_test_genre_50_2.jsonl \
    --val_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/whitebox_ir_test_genre_50_2.jsonl \
    --test_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/whitebox_ir_test_genre_50_2.jsonl \
    --reader mask \
    --mutation_source true \
    --mutation_target false \
    --labels all

 'test_avg_gen_time': 0.018045049038578252,
 'test_avg_loss': 0.4618696868419647,
 'test_avg_sari_avgaddscore': 0.32351193936580463,
 'test_avg_sari_avgdelscore': 0.38971423183258674,
 'test_avg_sari_avgkeepscore': 0.7176159622865126,
 'test_avg_sari_finalscore': 0.4769473778283013,
 'test_avg_summ_len': 31.577537351880473,
 'test_loss': tensor(0.4619, device='cuda:0'),
 'test_sari_avgaddscore': tensor(0.3235, device='cuda:0')}

"""
white_box_ir = {
    "keep": 0.7176159622865126,
    "delete": 0.38971423183258674,
    "add": 0.32351193936580463,
    "final": 0.4769473778283013,
}

# heuristic (ir)
"""
PYTHONPATH=src python3 -m error_correction.corrector.run \
    --model_name_or_path t5-base \
    --output_dir /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/output/masker/model=t5-base,lr=1e-4,masker=random,mutation_source=false,mutation_target=false,labels=all/seed=42 \
    --do_predict \
    --train_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/heuristic_ir_test_genre_50_2.jsonl \
    --val_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/heuristic_ir_test_genre_50_2.jsonl \
    --test_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/heuristic_ir_test_genre_50_2.jsonl \
    --reader mask \
    --mutation_source true \
    --mutation_target false \
    --labels all

 'test_avg_gen_time': 0.018235346368321567,
 'test_avg_loss': 0.8514469861984253,
 'test_avg_sari_avgaddscore': 0.2538436582171115,
 'test_avg_sari_avgdelscore': 0.7379545239704263,
 'test_avg_sari_avgkeepscore': 0.7423875461299191,
 'test_avg_sari_finalscore': 0.5780619094391524,
 'test_avg_summ_len': 32.136340598074,
 'test_loss': tensor(0.8514, device='cuda:0'),
 'test_sari_avgaddscore': tensor(0.2538, device='cuda:0')}

"""
heuristic_ir = {
    "keep": 0.7423875461299191,
    "delete": 0.7379545239704263,
    "add": 0.2538436582171115,
    "final": 0.5780619094391524,
}

# Percent difference for 1.0x10^{-4}

results = {
    "black-box (gold)": black_box_gold,
    "white-box (gold)": white_box_gold,
    "black-box (ir)": black_box_ir,
    "white-box (ir)": white_box_ir,
    "heuristic (ir)": heuristic_ir,
}
print(
    "\n 1.0x10^{-4}",
)
for result_key, value in results.items():
    paper_result = paper_results[result_key]
    for sari_key in value:
        percent_diff = (
            (value[sari_key] - paper_result[sari_key]) / paper_result[sari_key]
        ) * 100
        print(f"{result_key}: {sari_key}: {percent_diff:.3f}")


## lr 5.0x10^{-4} ##

# black-box (gold)
"""
PYTHONPATH=src python3 -m error_correction.corrector.run \
    --model_name_or_path t5-base \
    --output_dir /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/output/masker/model=t5-base,lr=5e-4,masker=random,mutation_source=false,mutation_target=false,labels=all/seed=42 \
    --do_predict \
    --train_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/blackbox_gold_test_genre_50_2_new.jsonl \
    --val_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/blackbox_gold_test_genre_50_2_new.jsonl \
    --test_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/blackbox_gold_test_genre_50_2_new.jsonl \
    --reader mask \
    --mutation_source true \
    --mutation_target false \
    --labels all

 'test_avg_loss': 0.6720343828201294,
 'test_avg_sari_avgaddscore': 0.304325203623293,
 'test_avg_sari_avgdelscore': 0.4369454204061377,
 'test_avg_sari_avgkeepscore': 0.6199115112798471,
 'test_avg_sari_finalscore': 0.45372737843642597,
 'test_avg_summ_len': 30.595370839668863,
 'test_loss': tensor(0.6720, device='cuda:0'),
 'test_sari_avgaddscore': tensor(0.3043, device='cuda:0')}

"""
black_box_gold = {
    "keep": 0.6199115112798471,
    "delete": 0.4369454204061377,
    "add": 0.304325203623293,
    "final": 0.45372737843642597,
}

# white-box (gold)
"""
PYTHONPATH=src python3 -m error_correction.corrector.run \
    --model_name_or_path t5-base \
    --output_dir /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/output/masker/model=t5-base,lr=5e-4,masker=random,mutation_source=false,mutation_target=false,labels=all/seed=42 \
    --do_predict \
    --train_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/whitebox_gold_test_genre_50_2.jsonl \
    --val_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/whitebox_gold_test_genre_50_2.jsonl \
    --test_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/whitebox_gold_test_genre_50_2.jsonl \
    --reader mask \
    --mutation_source true \
    --mutation_target false \
    --labels all

 'test_avg_gen_time': 0.01843100192496029,
 'test_avg_loss': 2.0821783542633057,
 'test_avg_sari_avgaddscore': 0.2755833033586278,
 'test_avg_sari_avgdelscore': 0.5681171542072248,
 'test_avg_sari_avgkeepscore': 0.603036164906422,
 'test_avg_sari_finalscore': 0.48224554082409155,
 'test_avg_summ_len': 32.88315584220789,
 'test_loss': tensor(2.0822, device='cuda:0'),
 'test_sari_avgaddscore': tensor(0.2756, device='cuda:0')}

"""
white_box_gold = {
    "keep": 0.603036164906422,
    "delete": 0.5681171542072248,
    "add": 0.2755833033586278,
    "final": 0.48224554082409155,
}

# black-box (ir)
"""
PYTHONPATH=src python3 -m error_correction.corrector.run \
    --model_name_or_path t5-base \
    --output_dir /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/output/masker/model=t5-base,lr=5e-4,masker=random,mutation_source=false,mutation_target=false,labels=all/seed=42 \
    --do_predict \
    --train_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/blackbox_ir_test_genre_50_2.jsonl \
    --val_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/blackbox_ir_test_genre_50_2.jsonl \
    --test_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/blackbox_ir_test_genre_50_2.jsonl \
    --reader mask \
    --mutation_source true \
    --mutation_target false \
    --labels all

 'test_avg_gen_time': 0.020487483896192957,
 'test_avg_loss': 2.384136199951172,
 'test_avg_sari_avgaddscore': 0.48997251628816063,
 'test_avg_sari_avgdelscore': 0.6048972252438273,
 'test_avg_sari_avgkeepscore': 0.4362633616721672,
 'test_avg_sari_finalscore': 0.5103777010680517,
 'test_avg_summ_len': 33.072295378259746,
 'test_loss': tensor(2.3841, device='cuda:0'),
 'test_sari_avgaddscore': tensor(0.4900, device='cuda:0')}

"""
black_box_ir = {
    "keep": 0.4362633616721672,
    "delete": 0.6048972252438273,
    "add": 0.48997251628816063,
    "final": 0.5103777010680517,
}

# white-box (ir)
"""
PYTHONPATH=src python3 -m error_correction.corrector.run \
    --model_name_or_path t5-base \
    --output_dir /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/output/masker/model=t5-base,lr=5e-4,masker=random,mutation_source=false,mutation_target=false,labels=all/seed=42 \
    --do_predict \
    --train_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/whitebox_ir_test_genre_50_2.jsonl \
    --val_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/whitebox_ir_test_genre_50_2.jsonl \
    --test_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/whitebox_ir_test_genre_50_2.jsonl \
    --reader mask \
    --mutation_source true \
    --mutation_target false \
    --labels all

 'test_avg_gen_time': 0.017810844776571775,
 'test_avg_loss': 0.8341274857521057,
 'test_avg_sari_avgaddscore': 0.2923924183485116,
 'test_avg_sari_avgdelscore': 0.386407456909216,
 'test_avg_sari_avgkeepscore': 0.6456012425847885,
 'test_avg_sari_finalscore': 0.44146703928083864,
 'test_avg_summ_len': 30.739824832560537,
 'test_loss': tensor(0.8341, device='cuda:0'),
 'test_sari_avgaddscore': tensor(0.2924, device='cuda:0')}


"""
white_box_ir = {
    "keep": 0.6456012425847885,
    "delete": 0.386407456909216,
    "add": 0.2923924183485116,
    "final": 0.44146703928083864,
}

# heuristic (ir)
"""
PYTHONPATH=src python3 -m error_correction.corrector.run \
    --model_name_or_path t5-base \
    --output_dir /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/output/masker/model=t5-base,lr=5e-4,masker=random,mutation_source=false,mutation_target=false,labels=all/seed=42 \
    --do_predict \
    --train_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/heuristic_ir_test_genre_50_2.jsonl \
    --val_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/heuristic_ir_test_genre_50_2.jsonl \
    --test_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/heuristic_ir_test_genre_50_2.jsonl \
    --reader mask \
    --mutation_source true \
    --mutation_target false \
    --labels all

 'test_avg_gen_time': 0.018401970904655007,
 'test_avg_loss': 1.2839473485946655,
 'test_avg_sari_avgaddscore': 0.22044634096987167,
 'test_avg_sari_avgdelscore': 0.7276056784616474,
 'test_avg_sari_avgkeepscore': 0.7333887944488076,
 'test_avg_sari_finalscore': 0.5604802712934421,
 'test_avg_summ_len': 32.75789829363068,
 'test_loss': tensor(1.2839, device='cuda:0'),
 'test_sari_avgaddscore': tensor(0.2204, device='cuda:0')}

"""
heuristic_ir = {
    "keep": 0.7333887944488076,
    "delete": 0.7276056784616474,
    "add": 0.22044634096987167,
    "final": 0.5604802712934421,
}

# Percent difference for 5.0x10^{-4}

results = {
    "black-box (gold)": black_box_gold,
    "white-box (gold)": white_box_gold,
    "black-box (ir)": black_box_ir,
    "white-box (ir)": white_box_ir,
    "heuristic (ir)": heuristic_ir,
}
print(
    "\n 5.0x10^{-4}",
)
for result_key, value in results.items():
    paper_result = paper_results[result_key]
    for sari_key in value:
        percent_diff = (
            (value[sari_key] - paper_result[sari_key]) / paper_result[sari_key]
        ) * 100
        print(f"{result_key}: {sari_key}: {percent_diff:.3f}")


# TRIGRAM

"""
PYTHONPATH=src python3 -m error_correction.corrector.run \
    --model_name_or_path t5-base \
    --output_dir /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/output/masker/model=t5-base,lr=5e-5,masker=random,mutation_source=false,mutation_target=false,labels=all/seed=42 \
    --do_predict \
    --train_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/trigram_test.jsonl \
    --val_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/trigram_test.jsonl \
    --test_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/trigram_test.jsonl \
    --reader mask \
    --mutation_source true \
    --mutation_target false \
    --labels all

 'test_avg_gen_time': 0.018401970904655007,
 'test_avg_loss': 1.2839473485946655,
 'test_avg_sari_avgaddscore': 0.22044634096987167,
 'test_avg_sari_avgdelscore': 0.7276056784616474,
 'test_avg_sari_avgkeepscore': 0.7333887944488076,
 'test_avg_sari_finalscore': 0.5604802712934421,
 'test_avg_summ_len': 32.75789829363068,
 'test_loss': tensor(1.2839, device='cuda:0'),
 'test_sari_avgaddscore': tensor(0.2204, device='cuda:0')}

"""


# NEW 40% RANDOM PER EPOCH

# black-box (gold)
"""
PYTHONPATH=src python3 -m error_correction.corrector.run \
    --model_name_or_path t5-base \
    --output_dir /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/ckpts/random-per-epoch \
    --do_predict \
    --train_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/blackbox_gold_test_genre_50_2_new.jsonl \
    --val_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/blackbox_gold_test_genre_50_2_new.jsonl \
    --test_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/blackbox_gold_test_genre_50_2_new.jsonl \
    --reader mask \
    --mutation_source true \
    --mutation_target false \
    --labels all

"""

# white-box (gold)
"""
PYTHONPATH=src python3 -m error_correction.corrector.run \
    --model_name_or_path t5-base \
    --output_dir /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/ckpts/random-per-epoch \
    --do_predict \
    --train_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/whitebox_gold_test_genre_50_2.jsonl \
    --val_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/whitebox_gold_test_genre_50_2.jsonl \
    --test_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/whitebox_gold_test_genre_50_2.jsonl \
    --reader mask \
    --mutation_source true \
    --mutation_target false \
    --labels all
"""

# black-box (ir)
"""
PYTHONPATH=src python3 -m error_correction.corrector.run \
    --model_name_or_path t5-base \
    --output_dir /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/ckpts/random-per-epoch \
    --do_predict \
    --train_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/blackbox_ir_test_genre_50_2.jsonl \
    --val_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/blackbox_ir_test_genre_50_2.jsonl \
    --test_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/blackbox_ir_test_genre_50_2.jsonl \
    --reader mask \
    --mutation_source true \
    --mutation_target false \
    --labels all

"""
# white-box (ir)
"""
PYTHONPATH=src python3 -m error_correction.corrector.run \
    --model_name_or_path t5-base \
    --output_dir /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/ckpts/random-per-epoch \
    --do_predict \
    --train_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/whitebox_ir_test_genre_50_2.jsonl \
    --val_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/whitebox_ir_test_genre_50_2.jsonl \
    --test_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/whitebox_ir_test_genre_50_2.jsonl \
    --reader mask \
    --mutation_source true \
    --mutation_target false \
    --labels all
"""

# heuristic (ir)
"""
PYTHONPATH=src python3 -m error_correction.corrector.run \
    --model_name_or_path t5-base \
    --output_dir /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/ckpts/random-per-epoch \
    --do_predict \
    --train_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/heuristic_ir_test_genre_50_2.jsonl \
    --val_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/heuristic_ir_test_genre_50_2.jsonl \
    --test_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/heuristic_ir_test_genre_50_2.jsonl \
    --reader mask \
    --mutation_source true \
    --mutation_target false \
    --labels all

"""



# NEW 50% RANDOM PER EPOCH

"""
PYTHONPATH=src python3 -m error_correction.corrector.run \
    --model_name_or_path t5-base \
    --output_dir /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/output/50_prob_masker \
    --do_predict \
    --train_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/blackbox_gold_test_genre_50_2_new.jsonl \
    --val_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/blackbox_gold_test_genre_50_2_new.jsonl \
    --test_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/blackbox_gold_test_genre_50_2_new.jsonl \
    --reader mask \
    --mutation_source true \
    --mutation_target false \
    --labels all

"""

# white-box (gold)
"""
PYTHONPATH=src python3 -m error_correction.corrector.run \
    --model_name_or_path t5-base \
    --output_dir /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/output/50_prob_masker \
    --do_predict \
    --train_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/whitebox_gold_test_genre_50_2.jsonl \
    --val_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/whitebox_gold_test_genre_50_2.jsonl \
    --test_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/whitebox_gold_test_genre_50_2.jsonl \
    --reader mask \
    --mutation_source true \
    --mutation_target false \
    --labels all
"""

# black-box (ir)
"""
PYTHONPATH=src python3 -m error_correction.corrector.run \
    --model_name_or_path t5-base \
    --output_dir /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/output/50_prob_masker \
    --do_predict \
    --train_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/blackbox_ir_test_genre_50_2.jsonl \
    --val_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/blackbox_ir_test_genre_50_2.jsonl \
    --test_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/blackbox_ir_test_genre_50_2.jsonl \
    --reader mask \
    --mutation_source true \
    --mutation_target false \
    --labels all

"""
# white-box (ir)
"""
PYTHONPATH=src python3 -m error_correction.corrector.run \
    --model_name_or_path t5-base \
    --output_dir /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/output/50_prob_masker \
    --do_predict \
    --train_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/whitebox_ir_test_genre_50_2.jsonl \
    --val_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/whitebox_ir_test_genre_50_2.jsonl \
    --test_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/whitebox_ir_test_genre_50_2.jsonl \
    --reader mask \
    --mutation_source true \
    --mutation_target false \
    --labels all
"""

# heuristic (ir)
"""
PYTHONPATH=src python3 -m error_correction.corrector.run \
    --model_name_or_path t5-base \
    --output_dir /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/output/50_prob_masker \
    --do_predict \
    --train_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/heuristic_ir_test_genre_50_2.jsonl \
    --val_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/heuristic_ir_test_genre_50_2.jsonl \
    --test_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/heuristic_ir_test_genre_50_2.jsonl \
    --reader mask \
    --mutation_source true \
    --mutation_target false \
    --labels all

"""

# NEW 60% RANDOM PER EPOCH

"""
PYTHONPATH=src python3 -m error_correction.corrector.run \
    --model_name_or_path t5-base \
    --output_dir /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/output/50_prob_masker \
    --do_predict \
    --train_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/blackbox_gold_test_genre_50_2_new.jsonl \
    --val_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/blackbox_gold_test_genre_50_2_new.jsonl \
    --test_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/blackbox_gold_test_genre_50_2_new.jsonl \
    --reader mask \
    --mutation_source true \
    --mutation_target false \
    --labels all

"""

# white-box (gold)
"""
PYTHONPATH=src python3 -m error_correction.corrector.run \
    --model_name_or_path t5-base \
    --output_dir /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/output/50_prob_masker \
    --do_predict \
    --train_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/whitebox_gold_test_genre_50_2.jsonl \
    --val_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/whitebox_gold_test_genre_50_2.jsonl \
    --test_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/whitebox_gold_test_genre_50_2.jsonl \
    --reader mask \
    --mutation_source true \
    --mutation_target false \
    --labels all
"""

# black-box (ir)
"""
PYTHONPATH=src python3 -m error_correction.corrector.run \
    --model_name_or_path t5-base \
    --output_dir /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/output/50_prob_masker \
    --do_predict \
    --train_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/blackbox_ir_test_genre_50_2.jsonl \
    --val_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/blackbox_ir_test_genre_50_2.jsonl \
    --test_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/blackbox_ir_test_genre_50_2.jsonl \
    --reader mask \
    --mutation_source true \
    --mutation_target false \
    --labels all

"""
# white-box (ir)
"""
PYTHONPATH=src python3 -m error_correction.corrector.run \
    --model_name_or_path t5-base \
    --output_dir /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/output/50_prob_masker \
    --do_predict \
    --train_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/whitebox_ir_test_genre_50_2.jsonl \
    --val_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/whitebox_ir_test_genre_50_2.jsonl \
    --test_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/whitebox_ir_test_genre_50_2.jsonl \
    --reader mask \
    --mutation_source true \
    --mutation_target false \
    --labels all
"""

# heuristic (ir)
"""
PYTHONPATH=src python3 -m error_correction.corrector.run \
    --model_name_or_path t5-base \
    --output_dir /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/output/50_prob_masker \
    --do_predict \
    --train_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/heuristic_ir_test_genre_50_2.jsonl \
    --val_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/heuristic_ir_test_genre_50_2.jsonl \
    --test_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/heuristic_ir_test_genre_50_2.jsonl \
    --reader mask \
    --mutation_source true \
    --mutation_target false \
    --labels all

"""