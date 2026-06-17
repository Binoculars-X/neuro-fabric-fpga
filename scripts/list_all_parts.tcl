set outfile [open "all_parts_list.txt" w]
set all_parts [get_parts]
foreach p $all_parts { 
    puts $outfile $p
}
close $outfile
puts "All [llength $all_parts] parts written"
exit
