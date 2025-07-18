SELECT * 
FROM dbo.application_test ATR
LEFT JOIN dbo.bureau_agg BA ON ATR.SK_ID_CURR = BA.SK_ID_CURR
LEFT JOIN dbo.pa_agg PA ON ATR.SK_ID_CURR = PA.SK_ID_CURR