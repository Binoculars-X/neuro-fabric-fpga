set outfile [open "first_100_parts.txt" w]
set all_parts [get_parts]
set count 0
foreach p $all_parts { 
    puts $outfile $p
    incr count
    if {$count >= 100} {
        break
    }
}
close $outfile
puts "First 100 parts written"
exit
