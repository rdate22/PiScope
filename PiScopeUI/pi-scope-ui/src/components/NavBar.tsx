import React from "react";
import { AppBar, Toolbar, IconButton, Typography, Stack, Button } from "@mui/material";
import HttpsSharp from "@mui/icons-material/HttpsSharp";

export const NavBar = () => {
    return (
        <AppBar position="static">
            <Toolbar>
                <IconButton size='large' edge = 'start' color ='inherit' aria-label="logo">
                    <HttpsSharp />
                </IconButton>
                <Typography variant="h6" component='div'>
                    SentriLock
                </Typography>
                <Stack direction= 'row' spacing={2}>
                    <Button color = 'inherit'> </Button>
                </Stack>
            </Toolbar>
        </AppBar>
    )
}