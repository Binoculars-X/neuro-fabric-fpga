set outfile [open "sample_parts.txt" w]
set all_parts [get_parts]
puts $outfile "Total parts available: [llength $all_parts]"
set count 0
foreach p $all_parts { 
    if {[string match "*zu9*" $p]} {
        puts $outfile $p
        incr count
    }
    if {$count >= 20} {
        break
    }
}
puts $outfile "Found $count matching parts"
close $outfile
puts "Sample parts written to sample_parts.txt"
exit
