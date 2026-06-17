set outfile [open "verify_zu9eg_install.txt" w]
set matches [get_parts -filter {NAME =~ *zu9eg*}]
puts $outfile "ZU9EG_COUNT=[llength $matches]"
foreach p $matches { puts $outfile $p }
set all_parts [get_parts]
puts $outfile "TOTAL_PARTS=[llength $all_parts]"
close $outfile
puts "Wrote verify_zu9eg_install.txt"
exit
