#!/bin/bash

# Parameters
PPT_DIR="./src/figures/1_1_powerpoint/"
TIF_DIR="./src/figures/1_2_tif/"
TIF_CROPPED_DIR="./src/figures/1_2_tif/cropped/"
PNG_DIR="./src/figures/1_3_png/"
LEGEND_DIR="./src/figures/2_legend/"
COMPILED_DIR="./src/figures/99_tex_compiled/"
ALL_COMPILED_PATH="./src/figures/99_tex_compiled/ALL_FIGURES.tex"

function main() {
    rm_templates
    if ! _main "$@"; then
        echo "An error occurred during the execution of _main."
        cp_templates
        _main
    fi
}


function _main(){
    tree ./src/figures
    if check_legend && (check_ppt || check_tif || check_png); then
        echo -e "\nFigures are being compiled..."
        mkdirs
        check_ppt && ppt2tif
        check_tif && crop_tif && tif2png

        if check_legend_png_consistency; then
            compile_legend
            gather_compiled
            handle_figure_visibility "$@"
        else
            return 1
        fi
    else
        echo -e "\nRequired legend files and either of ppt, tif, or png files are not present."
        exit 1
    fi
}

# function main() {
#     rm_templates
#     if ! _main; then
#         echo "aaa"
#         cp_templates && _main
#     fi
# }

function cp_templates() {
    echo -e "\nUsing preset templates..."            
    cp "${PPT_DIR}.Figure_ID_template.pptx" "${PPT_DIR}Figure_ID_template.pptx" -v
    cp "${TIF_DIR}.Figure_ID_template.tif" "${TIF_DIR}Figure_ID_template.tif" -v
    # cp "${PNG_DIR}.Figure_ID_template.png" "${PNG_DIR}Figure_ID_template.png" -v        
    cp "${LEGEND_DIR}.Figure_ID_template.tex" "${LEGEND_DIR}Figure_ID_template.tex" -v
}
    

function rm_templates() {
    rm "${PPT_DIR}Figure_ID_template.pptx" -f > /dev/null 2>&1
    rm "${TIF_DIR}Figure_ID_template.tif" -f > /dev/null 2>&1
    rm "${TIF_CROPPED_DIR}Figure_ID_template.tif" -f > /dev/null 2>&1    
    rm "${PNG_DIR}Figure_ID_template.png" -f > /dev/null 2>&1        
    rm "${LEGEND_DIR}Figure_ID_template.tex" -f > /dev/null 2>&1
}

function mkdirs() {
    mkdir -p "$LEGEND_DIR" "$PPT_DIR" "$TIF_DIR" "$PNG_DIR" "$COMPILED_DIR"
}

function check_ppt() {
    if compgen -G "$PPT_DIR*.pptx" > /dev/null; then
        return 0
    else
        echo -e "\nNo PowerPoint files found in $PPT_DIR."
        return 1
    fi
}

function ppt2tif() {
    ./scripts/sh/ppt2tif_all.sh
}

function check_tif() {
    if compgen -G "$TIF_DIR*.tif" > /dev/null; then
        return 0
    else
        echo -e "\nNo TIFF files found in $TIF_DIR."
        return 1
    fi
}

function crop_tif() {
    for tif_file in "$TIF_DIR"*.tif; do
        ./env/bin/python ./scripts/py/crop_tif.py -l "$tif_file"
    done
}

function tif2png() {
    for file in "$TIF_CROPPED_DIR"*.tif; do
        if [ -f "$file" ]; then
            local png_path="$PNG_DIR$(basename "$file" .tif).png"
            convert "$file" -resample 100x100 "$png_path"
        fi
    done

}
# function tif2png() {
#     for file in "$TIF_DIR"*.tif; do
#         if [ -f "$file" ]; then
#             convert "$file" -resample 100x100 "$PNG_DIR$(basename "$file" .tif).png"
#         fi
#     done
# }

function check_png() {
    if compgen -G "$PNG_DIR*.png" > /dev/null; then
        echo -e "\nPNG files are ready."
    else
        echo -e "\nNo PNG files found in $PNG_DIR."
    fi
}

function check_legend() {
    if compgen -G "$LEGEND_DIR*.tex" > /dev/null; then
        echo -e "\nLegend files are ready."
    else
        echo -e "\nNo legend files found in $LEGEND_DIR."
    fi
}

function check_legend_png_consistency() {
    local missing_counter=0
    local n_files=$(ls -A "$LEGEND_DIR"*.tex 2>/dev/null | wc -l)    

    if [ $n_files -gt 0 ]; then
        for legend_file in "$LEGEND_DIR"*.tex; do
            local base_name=$(basename "$legend_file" .tex)
            local corresponding_png="$PNG_DIR""$base_name".png

            if [ ! -f "$corresponding_png" ]; then
                echo -e "\nMissing PNG file for legend: $base_name"
                missing_counter=$((missing_counter + 1))
            fi
        done
    fi

    if [ $n_files -eq 0 ]; then
        echo -e "\nNo files are available under $LEGEND_DIR"
        return 1
    else
        if [ $missing_counter -eq 0 ]; then
            echo -e "\nAll legend files have corresponding PNG files."
            return 0
        else
            echo -e "\nMissing $missing_counter corresponding PNG files for legends."
            return 1
        fi
    fi
}

# function check_legend_png_consistency() {
#     local missing_counter=0

#     if legend_dir is not empty:
#     for legend_file in "$LEGEND_DIR"*.tex; do
#         echo $legend_file # ./src/figures/2_legend/*.tex
#         # local base_name=$(basename "$legend_file" .tex)
#         # local corresponding_png="$PNG_DIR""$base_name".png

#         # if [ ! -f "$corresponding_png" ]; then
#         #     echo -e "\nMissing PNG file for legend: $base_name"
#         #     missing_counter=$((missing_counter + 1))
#         # fi
#     done

#     if [ $missing_counter -eq 0 ]; then
#         echo -e "\nAll legend files have corresponding PNG files."
#     else
#         echo -e "\nMissing $missing_counter corresponding PNG files for legends."
#         return 1
#     fi
# }


function compile_legend() {
    rm "$COMPILED_DIR"/*.tex -f
    local ii=0
    for legend_file in "$LEGEND_DIR"Figure_ID_*.tex; do
        local fname=$(basename "$legend_file")
        local figure_id=${fname//Figure_ID_/}

        figure_id=${figure_id//.tex/}
        figure_id=$(echo -e "\n$figure_id" | tr '[:upper:]' '[:lower:]')
        local width=$(grep -oP '(?<=width=)[0-9.]+\\textwidth' "$legend_file")
        local tgt_file="$COMPILED_DIR$fname"

        if [ $ii -gt 0 ]; then
            echo -e "\n\\clearpage" >> "$tgt_file"
        fi
        ii=$((ii+1))
        
        cat <<EOF > "$tgt_file"
        \clearpage
        \begin{figure*}[ht]
            \pdfbookmark[2]{ID $figure_id}{figure_id_$figure_id}
            \centering
            \includegraphics[width=$width]{$PNG_DIR${fname%.tex}.png}
            $(cat "$legend_file")
            \label{fig:$figure_id}
        \end{figure*}
EOF
    done
}

function gather_compiled(){
    rm "$ALL_COMPILED_PATH" -f
	for fig_tex in "$COMPILED_DIR"*.tex; do
	    fname="${fig_tex%.tex}"
        echo "\input{${fname}}" >> "$ALL_COMPILED_PATH"
    done
    if [ -f "$ALL_COMPILED_PATH" ]; then
        echo -e "\n$ALL_COMPILED_PATH was successfully created."
    fi
}

function hide_figures() {
    for fig_tex in "$COMPILED_DIR"*.tex; do
        sed -i 's/^\\includegraphics/%\\includegraphics/g' "$fig_tex"
    done
}

function add_figures() {
    for fig_tex in "$COMPILED_DIR"*.tex; do
        sed -i 's/^% *\\includegraphics/\\includegraphics/g' "$fig_tex"
    done
}

function handle_figure_visibility() {
    if [ "$1" == "--no-figs" ]; then
        hide_figures
    else
        add_figures
    fi
}

_main "$@"

## EOF
