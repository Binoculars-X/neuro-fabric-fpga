foreach p [get_parts -filter {DISPLAY_NAME =~ *ZCU102*}] {
    puts $p
}
exit
