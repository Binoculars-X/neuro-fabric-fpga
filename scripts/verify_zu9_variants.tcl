set outfile [open "verify_zu9_variants.txt" w]
set matches [get_parts -filter {NAME =~ *zu9*}]
puts $outfile "ZU9_VARIANT_COUNT=[llength $matches]"
foreach p $matches { puts $outfile $p }
close $outfile
puts "Wrote verify_zu9_variants.txt"
exit
