#!/bin/bash

gen_compiled_tex_file() {
    main_file="./main.tex"
    output_file="./compiled.tex"
    cp "$main_file" "$output_file" -f

    echo

    substitute_input_command() {
        local file_path="$1"
        local temp_file=$(mktemp)

        while IFS= read -r line; do
            if [[ "$line" =~ \\input\{(.+)\} ]]; then
            # if [[ "$line" =~ ^[[:space:]]*[^%[:space:]]*\\input\{(.+)\} ]]; then                
                local input_path="${BASH_REMATCH[1]}.tex"
                if [[ -f "$input_path" ]]; then
                    echo "Processing $input_path"
                    cat "$input_path" >> "$temp_file"
                else
                    echo "Warning: File $input_path not found."
                    echo "$line" >> "$temp_file"
                fi
            else
                echo "$line" >> "$temp_file"
            fi
        done < "$file_path"

        mv "$temp_file" "$output_file"
    }

    # # Call substitute_input_command on the output file and repeat until no \input commands are left
    # while grep -q '\\input{' "$output_file"; do
    #     substitute_input_command "$output_file"
    # done
    # Call substitute_input_command on the output file and repeat until no \input commands are left
    while grep -q '^[^%]*\\input{' "$output_file"; do
        # echo $output_file
        substitute_input_command "$output_file"
    done

    echo -e "\nCompiled: $output_file"
}

gen_compiled_tex_file
# ./scripts/sh/gen_compiled.tex.sh

## EOF
