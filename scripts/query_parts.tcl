set outfile [open "all_zu9eg_parts.txt" w]
foreach p [get_parts -filter {NAME =~ *zu9eg*}] { 
    puts $outfile $p
}
close $outfile
puts "Parts written to all_zu9eg_parts.txt"
exit
