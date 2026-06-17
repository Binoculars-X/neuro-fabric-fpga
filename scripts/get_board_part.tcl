set outfile [open "board_to_part.txt" w]
set board [get_boards xilinx.com:zcu102:3.4]
set part_name [get_property PART_NAME $board]
puts $outfile "ZCU102 board part name: $part_name"
close $outfile
puts "Board part name retrieved"
exit
