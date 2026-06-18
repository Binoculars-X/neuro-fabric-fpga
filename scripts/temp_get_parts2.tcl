set outfile [open "parts_list.txt" w]
foreach p [get_parts -filter {NAME =~ *zu9eg*}] { 
    puts $outfile $p
}
close $outfile
exit
