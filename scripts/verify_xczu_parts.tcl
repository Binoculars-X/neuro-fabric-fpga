set outfile [open "verify_xczu_parts.txt" w]
set matches [get_parts -filter {NAME =~ xczu*}]
puts $outfile "XCZU_COUNT=[llength $matches]"
set shown 0
foreach p $matches {
  if {$shown < 20} { puts $outfile $p }
  incr shown
}
close $outfile
puts "Wrote verify_xczu_parts.txt"
exit
