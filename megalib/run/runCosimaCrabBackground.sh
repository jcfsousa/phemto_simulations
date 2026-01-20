mdelay cosima 22; cosima -z -v 0 /local/home/jf285468/documents/phd/phemto/phemto_simulations/megalib/sources/Background50-400keV_config4x4_1.5cm.source >> /dev/null & sleep 1;

mdelay cosima 22; cosima -z -v 0 /local/home/jf285468/documents/phd/phemto/phemto_simulations/megalib/sources/Crab_1Crab_50-400keV_config4x4_1.5cm.source >> /dev/null & sleep 1;
mdelay cosima 22; cosima -z -v 0 /local/home/jf285468/documents/phd/phemto/phemto_simulations/megalib/sources/Crab_100mCrab_50-400keV_config4x4_1.5cm.source >> /dev/null & sleep 1;
mdelay cosima 22; cosima -z -v 0 /local/home/jf285468/documents/phd/phemto/phemto_simulations/megalib/sources/Crab_10mCrab_50-400keV_config4x4_1.5cm.source >> /dev/null & sleep 1;
mdelay cosima 22; cosima -z -v 0 /local/home/jf285468/documents/phd/phemto/phemto_simulations/megalib/sources/Crab_1mCrab_50-400keV_config4x4_1.5cm.source >> /dev/null & sleep 1;
echo "Waiting for Cosima to run"
wait
