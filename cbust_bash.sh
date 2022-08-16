#!/bin/bash
fasta_file=$1
cd $2

module load Cluster-Buster/20220421-GCCcore-6.4.0
while read -r line
do
matrix_file=/staging/leuven/stg_00002/lcb/icistarget/data/motifCollection/v9/singletons/${line}.cb
cbust -c 0 -m 0 -f 5 ${matrix_file} ${fasta_file} > ${fasta_file%.fa}.${line}.bed
done < selected_motifs_$3.txt
cat ${fasta_file%.fa}.*.bed | grep -v "#" | awk '{if ($11 == "motif") print $0;}' |  cut -f1-6 > ${fasta_file%.fa}.selected.motif_$3.gff