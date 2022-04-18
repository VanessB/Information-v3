load "../common.gp"

set title "Взаимная информация (прямоугольники)"
set xlabel "I(X,Y)"
set ylabel "I(X,Y)"

set key left Left reverse

file_path = "data.csv" # data_path . "mutual_information/synthetic/"
compressed_file_path = "compressed.csv"

set output "plot" . extension

set style fill transparent solid 0.2 noborder
plot [*:*] file_path using 1:1 lc "red" lw 2 t "истинное значение", file_path using 1:($2 - $3):($2 + $3) with filledcurves lc "web-blue" notitle, file_path using 1:2 lc "blue" lw 2 t "оценка (без сжатия)", \
           compressed_file_path using 1:($2 - $3):($2 + $3) with filledcurves lc "web-green" notitle, compressed_file_path using 1:2 lc "green" lw 2 t "оценка (сжатие)"