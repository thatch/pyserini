# -------  Get Lucene Index   --------
DIR=nist/data/lucene-index-cord19-abstract-2020-05-01
TAR=lucene-index-cord19-abstract-2020-05-01.tar.gz

if [ ! -d "$DIR" ]; then
    wget -nc https://www.dropbox.com/s/wxjoe4g71zt5za2/${TAR}
    tar -xvzf ${TAR} -C nist/data
fi


# -------  Set Parameters   --------
ks=( 0 )
rs=( "2 1" )
alphas=( 0.57 )


# -------  Qrun   --------
mkdir -p runs/

for k in "${ks[@]}";do
for r in "${rs[@]}";do
for alpha in "${alphas[@]}";do
    time python nist/nist.py --k ${k} --R ${r} --alpha ${alpha}
    printf '\n'
done
done
done


# -------  Evaluation   --------
function score() {
    nist/trec_eval -c -M1000 -m all_trec nist/data/qrels_test.txt ${1} | grep 'ndcg_cut_10 '
    nist/trec_eval -c -M1000 -m all_trec nist/data/qrels_test.txt ${1} | grep 'map                   	all'
}

for f in runs/*k0*.txt;do
    echo $f
    echo "--------------------------------"
    score $f
    printf '\n'
done
