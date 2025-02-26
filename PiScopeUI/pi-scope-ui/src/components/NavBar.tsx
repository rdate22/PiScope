import React from "react";
import { AppBar, Toolbar, IconButton, Typography, Stack, Button, Theme } from "@mui/material";
import HttpsSharp from "@mui/icons-material/HttpsSharp";
import { Link } from "react-router-dom";

export const NavBar = () => {
    return (
        <AppBar position="fixed"
        sx ={{ top: 0, left: 0, width: '100%', zIndex: (theme: Theme) => theme.zIndex.drawer + 1 }}>
            <Toolbar>
                <IconButton size='large' edge = 'start' color ='inherit' aria-label="logo" component= {Link} to = "/">
                    <HttpsSharp />
                </IconButton>
                <Typography variant="h6" component='div'>
                    SentriLock
                </Typography>
                <Stack direction= 'row' spacing={2}>
                    <Button color = 'inherit' component={Link} to= "/live-feed"> View Live Feed </Button>
                </Stack>
            </Toolbar>
        </AppBar>
    )
}