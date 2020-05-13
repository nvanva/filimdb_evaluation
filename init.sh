task=$1
case "$task" in
  translit)
    mkdir TRANSLIT
    tar  -C 'TRANSLIT' -xvf TRANSLIT.tar.gz
  ;;
  *)
    tar xvf FILIMDB.tar.gz
    tar xvf PTB.tar.gz
  ;;
esac
