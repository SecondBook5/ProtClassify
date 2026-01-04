#!/bin/bash

# ------------- CONFIG -------------
INPUT_FASTA="pfam/query_eval.fasta"
DB="nr"
OUT_DIR="blastp_results"
TOP_HITS="blastp_top_hits.tsv"
UNIPROT_OUT="all_uniprot_annotations.tsv"
BATCH_SIZE=5
SLEEP_BETWEEN_BATCHES=20

# Fields from UniProt: Accession, Name, Protein Name, Organism, Length, Function, PDB IDs
UNIPROT_FIELDS="accession,id,protein_name,organism_name,length,comment(FUNCTION),xref_pdb"

# ------------- SETUP -------------
mkdir -p "$OUT_DIR/uniprot_metadata"
cd "$OUT_DIR" || exit 1
rm -f query_*.fasta *.tsv split_* accessions.txt "$TOP_HITS"

echo -e "query\tmatch\tpident\tlength\tevalue\tbitscore\tdescription" > "$TOP_HITS"

# ------------- SPLIT FASTA -------------
csplit -z -f split_ ../"$INPUT_FASTA" '/^>/' '{*}' >/dev/null

i=1
for f in split_*; do
    echo ">seq_$i" > "query_$i.fasta"
    tail -n +2 "$f" >> "query_$i.fasta"
    ((i++))
done

# ------------- RUN BLASTP IN BATCHES -------------
FORMAT="6 qseqid sseqid pident length evalue bitscore stitle"
batch=()
count=0

for q in query_*.fasta; do
    out="${q%.fasta}_blast.tsv"

    (
        echo "Running BLASTp on $q"
        blastp -query "$q" -db "$DB" -remote -outfmt "$FORMAT" -out "$out"
        if [[ $? -eq 0 ]]; then
            top_hit=$(head -n 1 "$out")
            echo -e "${q}\t${top_hit}" >> "$TOP_HITS"
        else
            echo -e "${q}\tBLAST_FAILED" >> "$TOP_HITS"
        fi
    ) &
    batch+=($!)
    ((count++))

    if (( count % BATCH_SIZE == 0 )); then
        echo "Waiting for batch of $BATCH_SIZE to finish..."
        wait "${batch[@]}"
        echo "Batch complete. Sleeping for $SLEEP_BETWEEN_BATCHES seconds."
        sleep "$SLEEP_BETWEEN_BATCHES"
        batch=()
    fi
done

# Wait for remaining processes
wait "${batch[@]}"

# ------------- EXTRACT ACCESSIONS FOR UniProt -------------
cut -f2 "$TOP_HITS" | grep -v 'BLAST_FAILED' | sed 's/.*|\(.*\)|.*/\1/' | sort -u > accessions.txt

# ------------- QUERY UniProt FOR ANNOTATIONS -------------
while read acc; do
    echo "Fetching UniProt data for $acc"
    curl -s "https://rest.uniprot.org/uniprotkb/${acc}.tsv?fields=${UNIPROT_FIELDS}" \
        -o "uniprot_metadata/${acc}.tsv"
done < accessions.txt

# ------------- MERGE UniProt ANNOTATIONS -------------
head -n 1 uniprot_metadata/*.tsv | head -n 1 > "$UNIPROT_OUT"
tail -n +2 -q uniprot_metadata/*.tsv >> "$UNIPROT_OUT"

echo -e "\nFinal annotation table: $OUT_DIR/$UNIPROT_OUT"
