%%
%   ASD Full Range Spectroradiometer Jump Correction
%   ------------------------------------------------
%   
%   Uses empirical correction coefficients to correct for temperature
%   related radiometric inter-channel steps.
%   For more information please see: 
%   Hueni, A. and Bialek, A. (2017). "Cause, Effect and Correction of Field Spectroradiometer Inter-channel Radiometric Steps." IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing 10(4): 1542-1551.
%
%   Please cite the above paper if you use this method in any published work.
%	
%   Input Parameters
%   ----------------
%   coeffs : correction coefficients supplied with this function (2nd order polynomial coefficients)
%
%   spectrum : spectrum to be corrected as a vector (2151 bands) -
%   dimension = 1 x 2151
%
%   wvl : wavelength vector (350 nm : 2500 nm) - dimension = 2151 x 1
%
%   iteration : used to control the iteration; set to zero upon call or leave empty []
%
%   jump_size_matrix : used in the iteration to collect jump size convergence data; leave empty [] upon call
%
%   interpolate_H2O: option to use when model appears noise restricted, as can be the case when applying it to reflectance data (e.g. white reference spectra)
%
%   Output Parameters
%   ----------------
%
%   corrected_spectrum : spectrum after correction
%
%   outside_T : estimated ambient temperature (currently biased due to
%   unresolved at-sensor radiance dependencies)
%
%   spec_corr_factors : applied correction factors (final factors after
%   all iterations)
%
%   jump_size_matrix : Matrix holding the jump sizes for VNIR-SWIR1 and
%   SWIR1-SWIR2 for all iterations
%
%
%   Example
%   --------
%           load('asd_temp_corr_coeffs.mat')
%           [corrected_spectrum, T_estimate, spec_corr_factors, jump_size_matrix] = ASD_Jump_Correction(asd_temp_corr_coeffs, spectrum_to_correct, wvl, [], []);
%
%
%           % example of applying a correction with H2O related noise suppression:
%           [corrected_spectrum_smoothed, T_estimate, spec_corr_factors, jump_size_matrix] = ASD_Jump_Correction(asd_temp_corr_coeffs, spectrum_to_correct, wvl, [], [], true);
%
%   ------------------------------------------------
%   (c) 2016-2021 A.Hueni, RSL, University of Zurich
%   ------------------------------------------------
%
%   Changes:
%   2017-05-16  Added handling for noise or model limited spectra
%   2017-06-09  Added handling for rare cases where two valid temperatures are found in quadratic model
%   2018-10-12  Added special handling of negative radiances in the VNIR,
%               presumably introduced by dark current correction issues in
%               the ASD instrument.
%   2020-07-14  Added dimension check and transpose if required
%   2021-02-15  Added a new option to interpolate the correction model in the H2O absorption bands. The current solution is to adopt
%               the parabolic correction as introduced in Beal, D. and Eamon, M. (2009). Preliminary Results of Testing and a Proposal for Radiometric Error Correction Using Dynamic, Parabolic Linear Transformations of “Stepped” Data (PCORRECT.EXE), Analytical Spectral Devices: 5.
%               This option is of interest when applying it to reflectance
%               data, in particular white reference spectra recordings.
	
%%

function [corrected_spectrum, outside_T, spec_corr_factors, jump_size_matrix, processing_notes] = ASD_Jump_Correction(coeffs, spectrum, wvl, iteration, jump_size_matrix, interpolate_H2O, iterations)

    spectrum_dim = size(spectrum);
    if spectrum_dim(1) ~= 1
        spectrum = spectrum';
    end
    
    wvl_dim = size(wvl);
    if wvl_dim(2) ~= 1
        wvl = wvl';
    end

    if isempty(iteration)
        iteration = 0;
    end

    if isempty(jump_size_matrix)
        jump_size_matrix = []; % used to document the convergence
    end
    
    if nargin() == 5
        interpolate_H2O = false;
    end
    
    if nargin() == 6
        iterations = 3;
    end
    
    processing_notes = {};
    
    
    % Special handling for water and other very dark targets that feature
    % negative radiances in the VNIR. Such problems may appear if the
    % radiance signals are very low and the dark current correction
    % (automatic process in the ASD) results in negative digital numbers.
    % It would appear that this could happen if the dark current fluctuates
    % slightly over time, or if there are some non-linearities for low
    % radiances.
    % The correction assumption is as follows: the dark current resulting in negative
    % DNs in the VNIR detector is a following the parabolic function.
    
    
    [m, vnir_range_start] = get_closest_wvl_index(wvl, 950);
    [m, vnir_range_end] = get_closest_wvl_index(wvl, 1000);
    [m, swir_range_start] = get_closest_wvl_index(wvl, 1001);
    [m, swir_range_end] = get_closest_wvl_index(wvl, 1050);
    
    vnir_spectrum = spectrum(1:vnir_range_end);
    negative_value_index = vnir_spectrum < 0;
    percent_of_negative_numbers = sum(negative_value_index) / length(negative_value_index) * 100;
    
    % Decide if a dark current issue exists.
    % The 1 percent threshold as chosen arbitrarily and may need adapting.
    % Typically, water spectra can have around 10% of negative radiance
    % band values in the VNIR.
    % Potentially, one could choose to correct every instance of negative
    % radiance, but more research into the matter is required first.
    if percent_of_negative_numbers > 1 && spectrum(vnir_range_end) < 0 && iteration == 0
        
        % this appears to be a case that needs dark current correcting
        % before a jump correction can be attempted.
        
        % Two methods present themselves:
        % a) Assume a fixed offset for all bands of the VNIR channel
        % b) Assume a parabolic offset
        
        % At this point the parabolic assumption is supported by
        % preliminary measurements. The correction method can chosen by a
        % flag.
        
        parabolic = 1;
        offset = 2;
        
        corr_method = parabolic;
        
        if corr_method == offset
            % Try to use minimum value to correct the spectrum.
            % Using the last band for correction is also not straightforward, as a
            % zero value leads to infinity when trying to calculate a
            % correction factor. Therefore, in case the last band value is the minimum,
            % then the standard deviation of the last two bands is added as correction factor.
            
            method = 'offset';
            
            [min_val, min_val_ind] = min(spectrum(1:vnir_range_end));
            
            if 1 == 0
                % documentation
                figure
                plot(wvl, spectrum);
                hold
                plot(wvl, [(spectrum(1:vnir_range_end) - min_val) spectrum(swir_range_start:end)], 'r' );
            end
            
            if(min_val_ind ~= vnir_range_end)
                % the last band is not the smallest value, hence, no
                % problem with the later correction
                spectrum(1:vnir_range_end) = spectrum(1:vnir_range_end) - min_val;
            else
                % the last band is the smallest value. It must be greater
                % than zero, hence, add the estimated noise of the last 2 bands to raise
                % above zero.
                spectrum(1:vnir_range_end) = spectrum(1:vnir_range_end) - min_val + (std(spectrum(vnir_range_end-1:vnir_range_end)));               
            end
            
        end
        
        if corr_method == parabolic
            % ASD Parabolic Correction
            method = 'parabolic';
            
            corr_factors = ones(size(spectrum));

            [m, i725] = get_closest_wvl_index(wvl, 725);
            [m, i1000] = get_closest_wvl_index(wvl, 1000);
            [m, i1001] = get_closest_wvl_index(wvl, 1001);
            %         [m, i1800] = get_closest_wvl_index(spectra.wvl, 1800);
            %         [m, i1950] = get_closest_wvl_index(spectra.wvl, 1950);
            %         [m, i1801] = get_closest_wvl_index(spectra.wvl, 1801);


            x = 350:2500;
            y = ((x-725).^2 .* (spectrum(i1001) - spectrum(i1000))) ./ (spectrum(i1000) * (1000 - 725)^2) + 1;

            corr_factors(i725:i1000) = y(i725:i1000);        

            if 1 == 0
                % documentation: comparison with offset method
                figure
                plot(wvl, spectrum);
                hold
                plot(wvl, [(spectrum(1:vnir_range_end) - min_val) spectrum(swir_range_start:end)], 'r' );
                plot(wvl, spectrum .* corr_factors, 'm')

                figure
                plot(wvl, corr_factors);
            end
            
            spectrum = spectrum .* corr_factors;
        
        end
        
        processing_notes{end+1} = ['ASD Jump Correction: Correction of VNIR channel for a presumed dark current issue (negative radiances) using ' method ' method'];
        dark_current_corrected = true;
        
        if 1 == 0
           % used to investigate the algorithms to detect negative radiances
           figure 
           plot(wvl(vnir_range_start:swir_range_end), spectrum(vnir_range_start:swir_range_end))

           figure
           plot(wvl(1:vnir_range_end), negative_value_index)
           title('Negative value index')

           [m ,first_negative_band_index] = min(wvl(1:vnir_range_end)'.*negative_value_index + negative_value_index);       

        end        
        
    end
    

    % use linear extrapolation to estimate the values in the last VNIR band and
    % first SWIR2 band
    [m, i] = get_closest_wvl_index(wvl, 1001);
    p = polyfit(1001:1003, spectrum(i:i+2), 1);
    last_vnir_estimate = polyval(p, 1000);
    
    if 1==0    
        % documentation        
        figure
        plot(1001:1003, spectrum(i:i+2))
        hold
        plot(1000, last_vnir_estimate, 'o')
        plot(997:1000, spectrum(i-4:i-1), 'r')
        
        legend('swir1', 'vnir 1 last band estimate', 'vnir')
    end    
    
    last_vnir_ind = i-1;
    first_swir = spectrum(i);
    last_vnir = spectrum(last_vnir_ind);
    jump_size_matrix(1,end+1) = first_swir - last_vnir;

    % catch the case that the last band is zero
    if(last_vnir > 0)       
        corr_factor_last_vnir = last_vnir_estimate / last_vnir;
    else
        % substitute the last band value with zero plus the noise of the
        % last two bands to get a positive correction factor
        corr_factor_last_vnir = last_vnir_estimate / std([spectrum(last_vnir_ind-1) last_vnir]);
    end
    
    [m, first_swir2_ind] = get_closest_wvl_index(wvl, 1801);
    last_swir1_ind = first_swir2_ind-1;
    p = polyfit(1798:1800, spectrum(last_swir1_ind-2:last_swir1_ind), 1);
    first_swir2_estimate = polyval(p, 1801);
    
    if 1==0    
        % documentation
        figure
        plot(1798:1800, spectrum(last_swir1_ind-2:last_swir1_ind))
        hold
        plot(1801, first_swir2_estimate, 'o')   
        plot(1801:1803, spectrum(first_swir2_ind:first_swir2_ind+2), 'r')
    end

    
    first_swir2 = spectrum(first_swir2_ind);
    last_swir1 = spectrum(last_swir1_ind);
    jump_size_matrix(2,end) = last_swir1 - first_swir2;

    corr_factor_first_swir2 = first_swir2_estimate / first_swir2;
    
    % goal: identify an outside temperature where the correction gains
    % established above are met
    % ax2 + bx + c = 0
    
    if 1 == 0
        % debugging: show the polynomial and the straight line of the
        % solutions to be found
        
        figure
        x=-100:200;
        y=coeffs(last_vnir_ind, 1)*x.^2 + coeffs(last_vnir_ind, 2)*x + coeffs(last_vnir_ind, 3);        
        plot(x, y);
        hold
        plot(x, ones(size(x))*corr_factor_last_vnir, 'r')
        title('2nd Order polynomial and correction factor last VNIR band');
        
    end
    
    
    [T_vnir, Ts_vnir, T_solution_vnir] = rqe(coeffs(last_vnir_ind, 1),coeffs(last_vnir_ind, 2),coeffs(last_vnir_ind, 3)-corr_factor_last_vnir);
    [T_swir, Ts_swir, T_solution_swir] = rqe(coeffs(first_swir2_ind, 1),coeffs(first_swir2_ind, 2),coeffs(first_swir2_ind, 3)-corr_factor_first_swir2);
    
    outside_T = mean([T_vnir T_swir]);
    
    % get transformation factors using these temperatures and correct the
    % spectrum.   
    [m, splice_band] = get_closest_wvl_index(wvl, 1726);
    T_vector = zeros(size(wvl));
    T_vector(1:splice_band) = T_vnir;
    T_vector(splice_band+1:end) = T_swir;
    
    spec_corr_factors = coeffs(:,1).*T_vector.^2 + coeffs(:,2).*T_vector + coeffs(:,3);
    
    if interpolate_H2O
        
        [~, swirsplit_ind] = get_closest_wvl_index(wvl, 1960);
        
        if 1 ==0
            % documentation
            figure
            plot(wvl, spec_corr_factors)
            hold
            plot(wvl, ones(size(wvl)), 'r')
            
            figure
            plot(wvl(1:last_vnir_ind), spec_corr_factors(1:last_vnir_ind))
            hold
            plot(wvl(1:last_vnir_ind), ones(size(wvl(1:last_vnir_ind))), 'r')      
            
            figure
            plot(wvl(last_vnir_ind+1:last_swir1_ind), spec_corr_factors(last_vnir_ind+1:last_swir1_ind))
            hold
            plot(wvl(last_vnir_ind+1:last_swir1_ind), ones(size(wvl(last_vnir_ind+1:last_swir1_ind))), 'r')   
            
            
            % full swir2
            figure
            plot(wvl(first_swir2_ind:end), spec_corr_factors(first_swir2_ind:end))
            hold
            plot(wvl(first_swir2_ind:end), ones(size(wvl(first_swir2_ind:end))), 'r')
     
            
            % swir2 part 1
            figure
            plot(wvl(first_swir2_ind:swirsplit_ind), spec_corr_factors(first_swir2_ind:swirsplit_ind))
            hold
            plot(wvl(first_swir2_ind:swirsplit_ind), ones(size(wvl(first_swir2_ind:swirsplit_ind))), 'r')         
            
            % swir2 part 1
            figure
            plot(wvl(swirsplit_ind+1:end), spec_corr_factors(swirsplit_ind+1:end))
            hold
            plot(wvl(swirsplit_ind+1:end), ones(size(wvl(swirsplit_ind+1:end))), 'r')              

        end
        
        % approach 1: carry out massive smoothing per detector
        spec_corr_factors_smoothed = spec_corr_factors;
        c = polyfit(wvl(1:last_vnir_ind), spec_corr_factors(1:last_vnir_ind), 3);

        spec_corr_factors_smoothed(1:last_vnir_ind) = polyval(c, wvl(1:last_vnir_ind));
        
        c = polyfit(wvl(last_vnir_ind+1:last_swir1_ind), spec_corr_factors(last_vnir_ind+1:last_swir1_ind), 3);

        spec_corr_factors_smoothed(last_vnir_ind+1:last_swir1_ind) = polyval(c, wvl(last_vnir_ind+1:last_swir1_ind));
        
        c = polyfit(wvl(first_swir2_ind:swirsplit_ind), spec_corr_factors(first_swir2_ind:swirsplit_ind), 3);

        spec_corr_factors_smoothed(first_swir2_ind:swirsplit_ind) = polyval(c, wvl(first_swir2_ind:swirsplit_ind));
        
        % avoid overfitting: only smooth in initial solution
        if iteration == 0
            c = polyfit(wvl(first_swir2_ind:swirsplit_ind), spec_corr_factors(first_swir2_ind:swirsplit_ind), 3);
            
            spec_corr_factors_smoothed(first_swir2_ind:swirsplit_ind) = polyval(c, wvl(first_swir2_ind:swirsplit_ind));
            
            c = polyfit(wvl(swirsplit_ind+1:end), spec_corr_factors(swirsplit_ind+1:end), 2);
            
            spec_corr_factors_smoothed(swirsplit_ind+1:end) = polyval(c, wvl(swirsplit_ind+1:end));
        end
        
        % 2: later SWIR2 part to be set to constant corr value of 1,
        % similar to the ASD parabolic correction
        spec_corr_factors_smoothed(swirsplit_ind:end) = ones(size(spec_corr_factors_smoothed(swirsplit_ind:end))) * spec_corr_factors_smoothed(swirsplit_ind);

        % 3: SWIR2 uses parabolic correction
        [m, i1800] = get_closest_wvl_index(wvl, 1800);
        [m, i1950] = get_closest_wvl_index(wvl, 1950);
        [m, i1801] = get_closest_wvl_index(wvl, 1801);
        
        y = ((wvl-1950).^2 .* (spectrum(i1800) - spectrum(i1801))) ./ (spectrum(i1801) * (1800 - 1950)^2) + 1;
        
        spec_corr_factors_smoothed(i1801:i1950) = y(i1801:i1950);
        spec_corr_factors_smoothed(i1950:end) = 1;
        
        
        % 4: VNIR parabolic solution
        [m, i725] = get_closest_wvl_index(wvl, 725);
        [m, i1000] = get_closest_wvl_index(wvl, 1000);
        [m, i1001] = get_closest_wvl_index(wvl, 1001);
        
        y = ((wvl-725).^2 .* (spectrum(i1001) - spectrum(i1000))) ./ (spectrum(i1000) * (1000 - 725)^2) + 1;
        
        spec_corr_factors_smoothed(i725:i1000) = y(i725:i1000);

        spec_corr_factors_smoothed(1:i725) = 1;
        
        % 5: SWIR 1 set to constant
        spec_corr_factors_smoothed(last_vnir_ind+1:last_swir1_ind) = 1;
        
        if 1 ==0
            % documentation
            
            figure
            plot(ones(size(spec_corr_factors_smoothed(first_swir2_ind:end)) * spec_corr_factors_smoothed(swirsplit_ind)));
            
            figure
            plot(wvl, spec_corr_factors)
            hold
            plot(wvl(1:last_vnir_ind), ones(size(wvl(1:last_vnir_ind))), 'r')                       
            plot(wvl, spec_corr_factors_smoothed, 'g', 'linewidth', 2.5)
            
        end
        
        spec_corr_factors = spec_corr_factors_smoothed;
        
    end
    
    % correction for model limits due to noise
    if ~T_solution_vnir
        spec_corr_factors(1:splice_band) = 1;
        disp('Attention: no VNIR correction due to noise or model limit');
        processing_notes{end+1} = 'ASD Jump Correction: No VNIR correction due to noise or model limit';
    end
    
    
    if ~T_solution_swir
        spec_corr_factors(splice_band+1:end) = 1;
        disp('Attention: no SWIR correction due to noise or model limit');
        processing_notes{end+1} = 'ASD Jump Correction: No SWIR correction due to noise or model limit';
    end
    
    
    corrected_spectrum = spectrum .* spec_corr_factors';
    
    
    if 1 ==0
        % documentation
        figure
        plot(wvl, spec_corr_factors)
        
        figure
        plot(wvl, corrected_spectrum, 'r*')
        hold
        plot(wvl, spectrum, 'b')
    end

    % iterative call
    if iteration < iterations
        [corrected_spectrum_iterated, ~, ~, jump_size_matrix] = ASD_Jump_Correction(coeffs, corrected_spectrum, wvl, iteration+1, jump_size_matrix, interpolate_H2O);

        if (1==0)
            % documentation
            figure
            hold
            plot(wvl, spectrum, 'k');
            plot(wvl, corrected_spectrum)
            plot(wvl, corrected_spectrum_iterated, 'r')
        end   
        
        corrected_spectrum = corrected_spectrum_iterated;
        
        % recalculate the temperatures based on iterated correction
        % coefficients
        
        spec_corr_factors_recalc = corrected_spectrum ./ spectrum;
        spec_corr_factors = spec_corr_factors_recalc;
        
        [T_vnir, Ts_vnir] = rqe(coeffs(last_vnir_ind, 1),coeffs(last_vnir_ind, 2),coeffs(last_vnir_ind, 3)-spec_corr_factors_recalc(last_vnir_ind));
        [T_swir, Ts_swir] = rqe(coeffs(first_swir2_ind, 1),coeffs(first_swir2_ind, 2),coeffs(first_swir2_ind, 3)-spec_corr_factors_recalc(first_swir2_ind));

        outside_T = mean([T_vnir T_swir]);     

    else
        
        if 1 == 0
            % documentation
            FontSize = 12;
            TitleFontSize = 16;
            linewidth = 1.5;    

            fh=figure;
            [AX,f1,f2] = plotyy(1:size(jump_size_matrix,2), jump_size_matrix(1,:), 1:size(jump_size_matrix,2), jump_size_matrix(2,:));

            set(f1, 'LineWidth',linewidth)
            set(f2, 'LineWidth',linewidth)

            legend({'VNIR Jump Size', 'SWIR2 Jump Size'}, 'Location', 'NorthEast');
            title_str = 'Convergence of Jump Sizes';
            title(title_str, 'FontSize', TitleFontSize);

            set(get(AX(1),'Ylabel'),'String','Radiance (L_\lambda) [W/m2/sr/nm]', 'FontSize', FontSize)
            set(get(AX(2),'Ylabel'),'String','Radiance (L_\lambda) [W/m2/sr/nm]', 'FontSize', FontSize)

            

            set(get(AX(1),'Xlabel'),'String','Iteration Step', 'FontSize', FontSize)
            set(get(AX(2),'Xlabel'),'String','Iteration Step', 'FontSize', FontSize)
            
            AX(1).XTick = 1:4;

            print_pdf(fh, '/Users/andyhueni/Data/Studies/RSL/Instruments/ASD/Temperature Experiment at NPL - Results/', title_str);

        end
        
            
    end

end


% Predict the temperature by finding the roots of the second order polynomial
% The two solutions are checked for a feasible
% range to return a single assumed outside temperature
function [T, Ts, T_solution] = rqe(a,b,c)
    Ts(1) = (-b + sqrt(b^2 - 4 * a * c))/(2*a);
    Ts(2) = (-b - sqrt(b^2 - 4 * a * c))/(2*a);

    T_solution = true;
    
    % temperature range: -10C - 70C : some sunglint effects can lead to
    % jump sizes that result in unreasonalbly high assumed temperatures:
    % these effects are essentially NOT temperature effects but field of
    % view issues!
    
    if isreal(Ts)
    
        T_feasible_ind = Ts > -10 & Ts < 70;
        
        if sum(T_feasible_ind) == 2
           % rare case of two temperatures being in the feasible range
           % select the smaller one in an absolute sense
           [m, T_feasible_ind] = min(abs(Ts));
            
        end
        

        T = Ts(T_feasible_ind);
        
        if isempty(T)
            % unrealistic temperatures have been found; this is likely due
            % to either a noise limitation or a FOV issue.
            T = 24.5;
            T_solution= false;
        end
        
        
    else
        
        % Complex solutions of the temperature equation indicate that the
        % presumed jump is too big to be accommodated by the model. The
        % likely reason is that the jump is lost in the sensor noise.
        % A cleaner solution would be to test the jump size versus NedL.
        
        % In this case we fallback to the assumed standard temperature
        % For the correction, no correction factors are calculated if a
        % complex solution is found.
        T = 24.5;        
        T_solution = false;
    end
    
end
