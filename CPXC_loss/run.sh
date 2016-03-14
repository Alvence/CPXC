#for file in anneal credit-a diabetes hepatitis ILPD iris labor sick vote vowel
for file in `ls data/*.arff`
#for file in "data/sick.arff"
do 
  for alg in svm nn nbc
  do
    for min_sup_ratio in 0.01 0.02 0.05 0.1 0.2
    do
      for ratio in 50 30 20 10 7 5
      do
        echo "./main -t temp -d $file -r $min_sup_ratio -l $ratio -a $alg"
        timeout 2000 ./main -t temp -d $file -r $min_sup_ratio -l $ratio -a $alg >> log/log.txt
     done
    done
  done
done

