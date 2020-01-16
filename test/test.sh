# ./diff.sh executable_path input_path output_path
diff $3 <(./$1 < $2)