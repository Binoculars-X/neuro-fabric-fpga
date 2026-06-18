set outfile [open "verify_zu9eg.txt" w]
foreach p [get_parts -filter {NAME =~ *zu9eg*}] {
    puts $outfile $p
}
close $outfile
puts "ZU9EG parts written to verify_zu9eg.txt"
exit
