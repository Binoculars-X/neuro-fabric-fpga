set outfile [open "zcu102_check.txt" w]
puts $outfile "=== BOARDS WITH ZCU102 ==="
foreach board [get_boards -filter {NAME =~ *zcu102*}] {
    puts $outfile $board
}
puts $outfile ""
puts $outfile "=== ALL BOARDS WITH ZU9 ==="
foreach board [get_boards -filter {NAME =~ *zu9*}] {
    puts $outfile $board
}
close $outfile
puts "Board search complete"
exit
