#!/bin/bash -l

cd out

DIRS="$(ls -d */)"

DIRS_ARR=($DIRS)

head -2 ${DIRS_ARR[0]}/neargrid/ACF.dat > charges_neargrid.txt
head -2 ${DIRS_ARR[0]}/weight/ACF.dat > charges_weight.txt

for DIR in $DIRS; do
  cd $DIR
  I_AT=$(echo $DIR | awk -F'_' '{ print $2 }')
  I_AT=${I_AT::-1}
  I_AT=$((I_AT+1))
  LINE_NR=$((I_AT+2))

  cd neargrid
  LINE="$(sed "${LINE_NR}q;d" ACF.dat)"
  echo "$LINE" >> ../../charges_neargrid.txt
  cd ..
  
  cd weight
  LINE="$(sed "${LINE_NR}q;d" ACF.dat)"
  echo "$LINE" >> ../../charges_weight.txt
  cd ..

  cd ..
done

echo "---- NEARGRID METHOD"
cat charges_neargrid.txt

echo ""
echo "---- WEIGHT METHOD"
cat charges_weight.txt

