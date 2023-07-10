REPO=$PWD
DIR=$REPO/download
mkdir -p $DIR

# download XNLI dataset
function download_xnli {
    OUTPATH=$DIR/xnli-tmp/
    if [ ! -d $OUTPATH/XNLI-MT-1.0 ]; then
      if [ ! -f $OUTPATH/XNLI-MT-1.0.zip ]; then
        wget -c https://dl.fbaipublicfiles.com/XNLI/XNLI-MT-1.0.zip -P $OUTPATH -q --show-progress
      fi
      unzip -qq $OUTPATH/XNLI-MT-1.0.zip -d $OUTPATH
    fi
    if [ ! -d $OUTPATH/XNLI-1.0 ]; then
      if [ ! -f $OUTPATH/XNLI-1.0.zip ]; then
        wget -c https://dl.fbaipublicfiles.com/XNLI/XNLI-1.0.zip -P $OUTPATH -q --show-progress
      fi
      unzip -qq $OUTPATH/XNLI-1.0.zip -d $OUTPATH
    fi
    python $REPO/utils_preprocess.py \
      --data_dir $OUTPATH \
      --output_dir $DIR/xnli/ \
      --task xnli
    rm -rf $OUTPATH
    echo "Successfully downloaded data at $DIR/xnli" >> $DIR/download.log
}

# download PAWS-X dataset
function download_pawsx {
    cd $DIR
    wget https://storage.googleapis.com/paws/pawsx/x-final.tar.gz -q --show-progress
    tar xzf x-final.tar.gz -C $DIR/
    python $REPO/utils_preprocess.py \
      --data_dir $DIR/x-final \
      --output_dir $DIR/pawsx/ \
      --task pawsx
    rm -rf x-final x-final.tar.gz
    echo "Successfully downloaded data at $DIR/pawsx" >> $DIR/download.log
}

download_xnli
download_pawsx
