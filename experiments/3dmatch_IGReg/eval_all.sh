# 25050 25051
for n in $(seq 39 39); do
    python test.py  --test_epoch=$n --benchmark=$1
    python eval.py  --test_epoch=$n --benchmark=$1 --method=lgr
    # for m in 250 500 1000 2500 5000; do
    # for m in 5000; do
    #     python eval.py  --test_epoch=$n --num_corr=$m --benchmark=$1 --method=ransac
    # done
    # python eval.py --test_epoch=$n --num_corr=250 --benchmark=$1 --method=svd
done

# for n in $(seq 26 26); do
#     for m in 250 500 1000 2500 5000; do
#         python eval.py --test_epoch=$n --num_corr=$m --benchmark=$1 --method=ransac
#     done
# done
# for x in 3DMatch 3DLoMatch; do
#     python test.py --test_epoch=$1 --benchmark=$x
#     python eval.py --test_epoch=$1 --benchmark=$x --method=lgr
# done




# for p in 24; do
#     for x in 3DMatch 3DLoMatch; do
#         python test.py --test_epoch=$p --benchmark=$x
#         for n in 250 500 1000 2500 5000; do
#             python eval.py --test_epoch=$p --num_corr=$n --benchmark=$x --method=ransac
#         done
#     done
# done