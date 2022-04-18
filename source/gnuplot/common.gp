reset

eps = 0
grayscale = 0

if (eps == 1){
set terminal eps enhanced font "liberation sans,16" fontscale 1.0 lw 4 size 12, 8
extension = ".eps"
title = 0
}
else{
set terminal pngcairo enhanced font "liberation sans,32" fontscale 1.0 size 1920, 1080
extension = ".png"
title = 1
}

#set datafile separator tab
set datafile separator ' '

set style increment default
set xrange  [ 0 : * ] noreverse writeback
set yrange  [ 0 : * ] noreverse writeback

#set key Left left top reverse

# Шаг делений.

set mxtics 10
set mytics 10

# Сетка.

set style line 100 lt 1 lc rgb "#444444" lw 1
set style line 101 lt 1 lc rgb "#CCCCCC" lw 1
set style line 102 lt 1 lc rgb "#EEEEEE" lw 1

set grid mytics ytics ls 101, ls 102
set grid mxtics xtics ls 101, ls 102

# Масштаб.

#set xtics 1
#set ytics 0.1
#set size ratio -10

# Палитра

#set palette gray positive gamma 1.5

if (grayscale == 1){
set linetype  1 lc rgb "#000000" lw 1
set linetype  2 lc rgb "#444444" lw 1
set linetype  3 lc rgb "#777777" lw 1
set linetype  4 lc rgb "#AAAAAA" lw 1
set linetype  5 lc rgb "#222222" lw 1
set linetype  6 lc rgb "#555555" lw 1
set linetype  7 lc rgb "#888888" lw 1
set linetype  8 lc rgb "#111111" lw 1
set linetype  9 lc rgb "#444444" lw 1
set linetype cycle  9
}

# Построение графиков.

set pointsize 2
set style data lines

set macro
small_points_common = "linespoints pointsize 1 lw 6"
points_common       = "linespoints pointsize 2 lw 6"
big_points_common   = "linespoints pointsize 4 lw 6"

plot_errbars      = "yerrorbars lw 6"
plot_small_points = small_points_common # . "pointtype 2"
plot_points       = points_common       # . "pointtype 2"
plot_big_points   = big_points_common   # . "pointtype 2"

data_path = '../data/'                                           # Директория с данными.