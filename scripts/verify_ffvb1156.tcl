set outfile [open "verify_ffvb1156.txt" w]
set matches [get_parts -filter {NAME =~ *ffvb1156*}]
puts $outfile "FFVB1156_COUNT=[llength $matches]"
foreach p $matches { puts $outfile $p }
close $outfile
puts "Wrote verify_ffvb1156.txt"
exit
