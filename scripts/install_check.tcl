set outfile [open "verify_install.txt" w]
set all_parts [get_parts]
puts $outfile "Total parts: [llength $all_parts]"
set count 0
foreach p $all_parts {
    if {[string match "*zu9eg*" $p]} {
        puts $outfile $p
        incr count
    }
}
if {$count == 0} {
    puts $outfile "NO ZU9EG PARTS FOUND"
}
close $outfile
puts "Check verify_install.txt"
exit
