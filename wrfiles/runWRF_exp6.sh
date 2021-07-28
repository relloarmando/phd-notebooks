#!/bin/bash
#   SODAR     LON 103.77  LAT 19.28, 
wrfdir=~/WRFV3/run
wpsdir=~/WPS
wps_run () {
 for pbl in 1 2 7; do
 sed -i "72s/[0-9],/${pbl},/g" $wrfdir/namelist.input
 sed -i "74s/[0-9],/${pbl},/g" $wrfdir/namelist.input
  for rad_p in 1 3 5; do
  echo "##### 09: Setting RAD=$rad_p and PBL=$pbl"
  sed -i "69s/[0-9],/${rad_p},/g" $wrfdir/namelist.input
  sed -i "70s/[0-9],/${rad_p},/g" $wrfdir/namelist.input
  cd; cd WPS/
  echo "##### 15: running WPS geogrid.exe"; ./geogrid.exe > ~/log.geogrid; sleep 2
  echo "##### 16: running WPS ungrib.exe";  ./ungrib.exe  > ~/log.ungrib; sleep 2
  echo "##### 17: running WPS metgrid.exe"; ./metgrid.exe > ~/log.metgrid; sleep 2
  wrf_run
  done
 done
	   }

wrf_run () {
 cd; cd $wrfdir; pwd
 echo "##### 24: Linking files from WPS folder"; ./link_met_em.sh
 echo "##### 25: Running real.exe"; ./real.exe > ~/log.realexe; sleep 1
 echo "##### 26: Running wrf.exe ${fname}_p${pbl}r${rad_p}"; ./wrf.exe   > ~/log.wrfexe; 	sleep 1
 outfile_move
           }
           
outfile_move () {
 mv wrfout_d01* /mnt/sda2/exp6_llj/20120816_d01_${fname}_p${pbl}r${rad_p}.nc 
 mv wrfout_d02* /mnt/sda2/exp6_llj/20120816_d02_${fname}_p${pbl}r${rad_p}.nc
 mv wrfout_d03* /mnt/sda2/exp6_llj/20120816_d03_${fname}_p${pbl}r${rad_p}.nc	
	}
	
# nam anl
echo "##### 37: Erasing old WRF files"
rm $wrfdir/met_em* $wrfdir/wrfinput_* $wrfdir/wrfout_d* $wpsdir/FILE:* $wpsdir/GRIBFILE* $wpsdir/met_em*
echo "##### 35: Linking grib files and grib tables"; cd; cd WPS/; 
./link_grib.csh ~/grib/201208_namanl/namanl_218_201208*; ln -sf ungrib/Variable_Tables/Vtable.NAM Vtable
fname=nam32; echo "45: Dataset =" $fname; metglevs=40
sed -i "53s/[0-9][0-9],/${metglevs},/g" $wrfdir/namelist.input
wps_run

# gfs anl3
echo "##### 46: Erasing old WRF files"
rm $wrfdir/met_em* $wrfdir/wrfinput_* $wrfdir/wrfout_d* $wpsdir/FILE:* $wpsdir/GRIBFILE* $wpsdir/met_em*
echo "##### 48: Linking grib files and grib tables"; cd; cd WPS/; 
./link_grib.csh ~/grib/201208_gfsanl3/gfsanl_3_201208*; ln -sf ungrib/Variable_Tables/Vtable.GFS Vtable
fname=gfs05; echo "34: Dataset =" $fname; metglevs=27
sed -i "53s/[0-9][0-9],/${metglevs},/g" $wrfdir/namelist.input
wps_run

#mail -s "Run WRF Update ${fname}_p${param}_r${rad_p}" ssrello@gmail.com <<< "DONE: ${fname}_p${param}_r${rad_p}"
