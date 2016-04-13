for file in "data/iris.arff"
#for file in "data/sick.arff"
do 
  for alg in svm nn nbc
  do
    for min_sup_ratio in 0.01 0.02 0.05 0.1 0.2
    do
      for ratio in 50 30 20 10 7 5
      do
        echo "./main -t temp -d $file -r $min_sup_ratio -l $ratio -a $alg"
        ./main -t temp -d $file -r $min_sup_ratio -l $ratio -a $alg >> new/log-iris.txt
     done
    done
  done
done
