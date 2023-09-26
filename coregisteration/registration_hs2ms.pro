PRO registration_hs2ms,mode,date,mstime,hstime,block
  COMPILE_OPT IDL2

  e = ENVI(/HEADLESS)

  top_path = '/Volumes/vino/rit_research/grapes/'

  ms_path = top_path+strmid(date,0,4)+$
  '/micasense/'+date+'/'+mstime+'_'+block+'/'
  ms_file = ms_path+mstime+'_'+block+'_band_stack.tif'

  hs_path = top_path+strmid(date,0,4)+$
  '/'+mode+'/'+date+'/'+hstime+'/radiance/'
  hs_files = FILE_SEARCH(hs_path+'*.hdr')

  IF (mode EQ 'vnir') THEN BEGIN
      ms_band = 1
      hs_band = 73
  ENDIF ELSE BEGIN
      ms_band = 4
      hs_band = 13
  ENDELSE

  ms_raster = e.OpenRaster(ms_file)
  ms_green = ENVISubsetRaster(ms_raster, BANDS=ms_band)

  FOR i=0,n_elements(hs_files)-1 DO BEGIN

      hs_file = hs_files[i]
      hs_raster = e.OpenRaster(hs_file)
      hs_green = ENVISubsetRaster(hs_raster, BANDS=hs_band)

      tp_file = strmid(hs_file,0,strlen(hs_file)-4)+'.pts'
      hs_file_out = strmid(hs_file,0,strlen(hs_file)-4)+'_reg'

      Task = ENVITask('GenerateTiePointsByCrossCorrelation')
      Task.INPUT_RASTER1 = ms_green
      Task.INPUT_RASTER2 = hs_green
      Task.MATCHING_WINDOW = 40
      Task.MINIMUM_MATCHING_SCORE = 0.8
      Task.Execute

      TiePoints = Task.OUTPUT_TIEPOINTS

      FilterTask = ENVITask('FilterTiePointsByGlobalTransform')
      FilterTask.INPUT_TIEPOINTS = TiePoints
      FilterTask.OUTPUT_TIEPOINTS_URI = tp_file
      FilterTask.Execute

      TiePoints2 = ENVITiePointSet(tp_file, INPUT_RASTER1=ms_green, INPUT_RASTER2=hs_raster)

      RegistrationTask = ENVITask('ImageToImageRegistration')
      RegistrationTask.DATA_IGNORE_VALUE = 0
      RegistrationTask.INPUT_TIEPOINTS = TiePoints2
      ;RegistrationTask.WARPING = 'Triangulation'
      RegistrationTask.OUTPUT_RASTER_URI = hs_file_out
      RegistrationTask.Execute

  ENDFOR

END