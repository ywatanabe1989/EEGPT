#!/bin/bash

# Parameters
CSV_DIR="./src/tables/1_1_csv/"
EXCEL_DIR="./src/tables/1_2_excel/"
LEGEND_DIR="./src/tables/2_legend/"
COMPILED_DIR="./src/tables/99_tex_compiled/"
ALL_COMPILED_PATH="./src/tables/99_tex_compiled/ALL_TABLES.tex"

# main() {
#     rm_templates
#     if ! _main "$@"; then
#         echo "An error occurred during the execution of _main."
#         cp_templates
#         echo "Exiting after failure. Please check the logs or error messages above for details."
#         # exit 1
#     fi
# }
main() {
    rm_templates
    if ! _main "$@"; then
        echo "An error occurred during the execution of _main."
        cp_templates
        _main
    fi
}

_main(){
    tree ./src/tables
    if check_legend && (check_excel || check_csv); then
        echo -e "\nTables are being compiled..."
        mkdirs

        # check_csv
        if ! check_csv; then
            check_excel && excel2csv
        fi
        check_csv

        if check_legend_csv_consistency; then
            compile_legend
            gather_compiled
        else
            return 1
        fi
    else
        echo -e "\nRequired legend files and either of excel or csv files are not present."
        return 1
    fi
}


rm_templates() {
    rm "${CSV_DIR}Table_ID_template.csv" -f > /dev/null 2>&1
    rm "${EXCEL_DIR}Table_ID_template.xlsx" -f > /dev/null 2>&1    
    rm "${LEGEND_DIR}Table_ID_template.tex" -f > /dev/null 2>&1
    rm "${COMPILED_DIR}Table_ID_template.tex" -f > /dev/null 2>&1    
}

cp_templates() {
    echo -e "\nUsing preset templates..."
    cp "${EXCEL_DIR}.Table_ID_template.xlsx" "${EXCEL_DIR}Table_ID_template.xlsx" -v    
    cp "${CSV_DIR}.Table_ID_template.csv" "${CSV_DIR}Table_ID_template.csv" -v
    cp "${LEGEND_DIR}.Table_ID_template.tex" "${LEGEND_DIR}Table_ID_template.tex" -v
}

mkdirs() {
    mkdir -p "$LEGEND_DIR" "$CSV_DIR" "$EXCEL_DIR" "$COMPILED_DIR"
}

check_excel() {
    if compgen -G "$EXCEL_DIR*.pptx" > /dev/null; then
        return 0
    else
        echo -e "\nNo Excel files found in $EXCEL_DIR."
        return 1
    fi
}

excel2csv() {
    for f in "$EXCEL_DIR"*.xlsx; do # [REVISED]
        ./scripts/py/excel2csv.py "$f" "${f%.xlsx}.csv" # [REVISED]
    done
}

check_csv() {
    if compgen -G "$CSV_DIR*.csv" > /dev/null; then
        return 0
    else
        echo -e "\nNo CSVF files found in $CSV_DIR."
        return 1
    fi
}

check_legend() {
    if compgen -G "$LEGEND_DIR*.tex" > /dev/null; then
        echo -e "\nLegend files are ready."
    else
        echo -e "\nNo legend files found in $LEGEND_DIR."
    fi
}

check_legend_csv_consistency() {
    local missing_counter=0
    local n_files=$(ls -A "$LEGEND_DIR"*.tex 2>/dev/null | wc -l)    

    if [ $n_files -gt 0 ]; then
        for legend_file in "$LEGEND_DIR"*.tex; do
            local base_name=$(basename "$legend_file" .tex)
            local corresponding_csv="$CSV_DIR""$base_name".csv

            if [ ! -f "$corresponding_csv" ]; then
                echo -e "\nMissing CSV file for legend: $base_name"
                missing_counter=$((missing_counter + 1))
            fi
        done
    fi

    if [ $n_files -eq 0 ]; then
        echo -e "\nNo files are available under $LEGEND_DIR"
        return 1
    else
        if [ $missing_counter -eq 0 ]; then
            echo -e "\nAll legend files have corresponding CSV files."
            return 0
        else
            echo -e "\nMissing $missing_counter corresponding CSV files for legends."
            return 1
        fi
    fi
}

compile_legend() {
    rm "$COMPILED_DIR"*.tex -f
    local ii=0
    for legend_file in "$LEGEND_DIR"Table_ID_*.tex; do
        local fname=$(basename "$legend_file")
        local table_id=${fname//Table_ID_/}
        table_id=${table_id//.tex/}
        table_id=$(echo -e "\n$table_id" | tr '[:upper:]' '[:lower:]')
        # local width=$(grep -oP '(?<=width=)[0-9.]+\\textwidth' "$legend_file")
        local tgt_file="$COMPILED_DIR$fname"

        if [ $ii -gt 0 ]; then
            echo -e "\n\\clearpage" >> "$tgt_file"
        fi

        ii=$((ii+1))
        # Determine the number of columns in the CSV file
        local csv_file="${CSV_DIR}Table_ID_${table_id}.csv" # [REVISED]
        csv_file=$(echo $csv_file | tr -d ' ') # [REVISED]
        out_file=${LEGEND_DIR}Table_ID_${table_id}
        out_file=$(echo $out_file | tr -d ' ') # [REVISED]        
        
        # local csv_file="$CSV_DIR"Table_ID_${table_id}.csv
        # echo $csv_file
        local num_columns=$(head -n 1 "$csv_file" | awk -F, '{print NF}')
        
        # # Determine the number of columns in the CSV file
        # num_columns=$(head -n 1 "$csv_file" | awk -F, '{print NF}')

        fontsize="\\small"
        
        # Create the LaTeX document
        {
            echo "\\pdfbookmark[2]{ID ${table_id}}{id_${table_id}}"
            echo "\\begin{table*}[htbp]"
            echo "\\centering"
            echo "$fontsize"
            echo "\\begin{tabular}{*{$num_columns}{c}}"
            echo "\\toprule"
            # Header
            head -n 1 "$csv_file" | {
                IFS=',' read -ra headers
                for header in "${headers[@]}"; do
                    echo -n "\\textbf{\\thead{${header}}}" # [REVISED]
                    if [[ $header != ${headers[-1]} ]]; then
                        echo -n " &" # [REVISED]
                    fi
                done
                echo "\\\\"
            }
            echo "\\midrule"

            # Table contents
            awk 'BEGIN {FPAT = "([^,]+)|(\"[^\"]+\")"; OFS=" & "; row_count=0} 
                NR>1 {
                    if (row_count % 2 == 1) {print "\\rowcolor{lightgray}"}
                    $1=$1; print $0"\\\\"
                    row_count++
                }' "$csv_file"            
            echo "\\bottomrule"
            echo "\\end{tabular}"
            echo "\\captionsetup{width=.${width}\\textwidth}" # [REVISED]
            echo "\\input{$out_file}"
            echo ""
            echo "\\label{tab:${table_id}}"            
            echo "\\end{table*}"
            echo "\\restoregeometry"

        } > "$tgt_file" # [REVISED]
    done
}



gather_compiled(){
    rm "$ALL_COMPILED_PATH" -f
	for tab_tex in "$COMPILED_DIR"*.tex; do
	    fname="${tab_tex%.tex}"
        echo "\input{${fname}}" >> "$ALL_COMPILED_PATH"
    done
    if [ -f "$ALL_COMPILED_PATH" ]; then
        echo -e "\n$ALL_COMPILED_PATH was successfully created."
    fi
}

main "$@"

# ## EOF
