for n in $(git ls-files)
do
    # echo "Working on $n file name now"
    # do something on $n below, say count line numbers
    CNT_FN = $(wc -l "$n")
    echo "$CNT_FN"
    read -ra ADDR <<< "$CNT_FN" #reading str as an array as tokens separated by IFS
    echo "$CNT_FN[0]"
done
