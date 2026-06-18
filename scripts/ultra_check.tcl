set outfile [open "available_ultra.txt" w]
foreach p [get_parts -filter {NAME =~ *ultrascale*}] {
    puts $outfile $p
}
close $outfile
puts "UltraScale devices written"
exit
